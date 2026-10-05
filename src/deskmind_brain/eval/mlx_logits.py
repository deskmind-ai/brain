"""MLX backend for the logit baseline: the fast path on Apple Silicon.

One request = one shared prefix (system prompt + state) and many short question suffixes ("branches").
The prefix is prefilled once. Branches are then scored in chunks without copying the prefix's attention KV:

- linear-attention (Gated DeltaNet) layers run the chunk as a batch; their per-sequence state is small
  (conv window + recurrent matrix), so it is simply repeated per branch;
- full-attention layers flatten the chunk into one sequence of C*S tokens that attends to the single copy of
  the prefix KV plus its own branch (block-causal mask), with RoPE offsets continuing from the prefix.

Right padding needs no mask: a branch's last real token never attends to the padding after it.
The prefix and its cache are kept for the next call with the same prefix (tournament rounds).
"""

from __future__ import annotations

import os
import shutil
import time
from collections import OrderedDict
from pathlib import Path

from deskmind_brain.eval.logits_base import LogitsPredictorBase, Prompt, find_format_file, shared_prefix_len

MLX_CACHE_DIR = Path.home() / ".cache" / "deskmind-brain" / "mlx"
# Upper bound for the attention scores of one flattened chunk (heads * queries * keys * 4 bytes).
ATTENTION_BUDGET_BYTES = 1 << 30
MAX_CHUNK = 64
# Format-3 prefix checkpoints kept across requests (one per recent task: system prompt + goal + rules).
CHECKPOINTS = 4


def resolve_model_path(model_id: str, bits: int | None) -> str:
    """A local directory for `model_id` (a path, or a hub id downloaded on first use), quantized to `bits` if given.

    Hub ids come from the Hugging Face Hub; ModelScope is the fallback when that fails, or first with
    DESKMIND_MODELSCOPE=1 (for networks where huggingface.co is slow or blocked)."""
    path = model_id
    if not Path(model_id).exists():
        path = None
        if os.environ.get("DESKMIND_MODELSCOPE") != "1":
            try:
                from huggingface_hub import snapshot_download

                path = snapshot_download(model_id)
            except Exception as e:  # noqa: BLE001 — fall back to ModelScope below
                print(f"huggingface_hub download of {model_id} failed ({type(e).__name__}); trying ModelScope")
        if path is None:
            from modelscope import snapshot_download

            path = snapshot_download(model_id)
    if not bits:
        return path
    out = MLX_CACHE_DIR / f"{model_id.replace('/', '--')}-q{bits}"
    if not (out / "config.json").exists():
        from mlx_lm import convert

        MLX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        convert(hf_path=path, mlx_path=str(out), quantize=True, q_bits=bits, q_group_size=64)
    # the quantized copy must carry the prompt-format flags the weights were trained with
    fmt = find_format_file(path)
    if fmt is not None and find_format_file(out) is None:
        shutil.copy(fmt, out / fmt.name)
    return str(out)


class MLXLogitsPredictor(LogitsPredictorBase):
    def __init__(self, model_id: str, bits: int | None = None, memory_limit_gb: float = 20.0):
        import mlx.core as mx
        from mlx_lm import load

        self.mx = mx
        self.name = f"mlx:{model_id}" + (f"@{bits}bit" if bits else "")
        # Hard ceiling on Metal allocations so an oversized request fails instead of freezing the machine.
        mx.set_memory_limit(int(memory_limit_gb * 2**30))
        mx.set_cache_limit(4 * 2**30)
        resolved = resolve_model_path(model_id, bits)
        model, self.tokenizer = load(resolved)
        self.lm = getattr(model, "language_model", model)
        self.text = self.lm.model
        self._init_labels(model_id, model_dir=resolved)
        self._prefix_key: tuple[int, ...] | None = None
        self._prefix_cache = None
        self._checkpoints: OrderedDict[tuple[int, ...], list] = OrderedDict()

    def _release_memory(self) -> None:
        self._prefix_key, self._prefix_cache = None, None
        self.mx.clear_cache()

    # ------------------------------------------------------------------ prefix

    def _copy_cache(self, cache):
        mx = self.mx
        out = self.lm.make_cache()
        for dst, src in zip(out, cache):
            dst.state = [mx.array(a) for a in src.state]
        mx.eval([c.state for c in out])
        return out

    def _prefill(self, prefix: list[int], head_len: int = 0):
        """Prefill the shared prefix of a request. With format 3, the part up to the state (system prompt + goal +
        rules) is kept as a checkpoint across requests, so later steps of the same task only prefill their state."""
        key = tuple(prefix)
        stats = self.stats
        if key == self._prefix_key:
            if stats is not None:
                stats["prefix_reused"] += 1
            return self._prefix_cache
        mx = self.mx
        start = time.perf_counter()
        head = tuple(prefix[:head_len]) if 0 < head_len < len(prefix) else None
        hit = head is not None and head in self._checkpoints
        if stats is not None:
            stats["prefills"] += 1
            stats["prefix_tokens"] = max(stats["prefix_tokens"], len(prefix))
            stats["prefilled_tokens"] += len(prefix) - (len(head) if hit else 0)
            if head is not None:
                stats["head_tokens"] = max(stats["head_tokens"], len(head))
                if stats["checkpoint"] is None:
                    stats["checkpoint"] = "hit" if hit else "miss"
        if hit:
            self._checkpoints.move_to_end(head)
            cache = self._copy_cache(self._checkpoints[head])
            rest = prefix[head_len:]
        else:
            cache = self.lm.make_cache()
            rest = prefix
            if head is not None:
                self.text(mx.array(prefix[:head_len])[None], cache=cache)
                mx.eval([c.state for c in cache])
                self._checkpoints[head] = self._copy_cache(cache)
                while len(self._checkpoints) > CHECKPOINTS:
                    self._checkpoints.popitem(last=False)
                rest = prefix[head_len:]
        self.text(mx.array(rest)[None], cache=cache)
        mx.eval([c.state for c in cache])
        self._prefix_key, self._prefix_cache = key, cache
        if stats is not None:
            stats["prefill_s"] += time.perf_counter() - start
        return cache

    # ------------------------------------------------------------------ branches

    def _branch_attention(self, layer, x, prefix_kv, offset: int):
        mx = self.mx
        attn = layer.self_attn
        h = layer.input_layernorm(x)
        C, S, _ = h.shape
        n_heads, n_kv = attn.num_attention_heads, attn.num_key_value_heads
        queries, gate = mx.split(attn.q_proj(h).reshape(C, S, n_heads, -1), 2, axis=-1)
        gate = gate.reshape(C, S, -1)
        queries = attn.q_norm(queries).transpose(0, 2, 1, 3)
        keys = attn.k_norm(attn.k_proj(h).reshape(C, S, n_kv, -1)).transpose(0, 2, 1, 3)
        values = attn.v_proj(h).reshape(C, S, n_kv, -1).transpose(0, 2, 1, 3)
        queries = attn.rope(queries, offset=offset)
        keys = attn.rope(keys, offset=offset)

        def flatten(t):  # (C, heads, S, d) -> (1, heads, C*S, d); token index = branch * S + position
            return t.transpose(1, 0, 2, 3).reshape(1, t.shape[1], C * S, t.shape[3])

        pk, pv = prefix_kv
        all_keys = mx.concatenate([pk, flatten(keys)], axis=2)
        all_values = mx.concatenate([pv, flatten(values)], axis=2)
        pos = mx.arange(S)
        branch = mx.arange(C)
        same_branch = branch[:, None, None, None] == branch[None, None, :, None]  # (C,1,C,1)
        causal = pos[None, :, None, None] >= pos[None, None, None, :]  # (1,S,1,S)
        block = (same_branch & causal).reshape(C * S, C * S)
        mask = mx.concatenate([mx.ones((C * S, offset), dtype=mx.bool_), block], axis=1)
        out = mx.fast.scaled_dot_product_attention(flatten(queries), all_keys, all_values, scale=attn.scale, mask=mask)
        out = out.reshape(n_heads, C, S, -1).transpose(1, 2, 0, 3).reshape(C, S, -1)
        hidden = x + attn.o_proj(out * mx.sigmoid(gate))
        return hidden + layer.mlp(layer.post_attention_layernorm(hidden))

    def _branch_logits(self, cache, suffixes: list[list[int]], offset: int):
        """Next-token logits at the last real token of each branch, shape (C, vocab)."""
        from mlx_lm.models.cache import ArraysCache

        mx = self.mx
        C, S = len(suffixes), max(len(s) for s in suffixes)
        ids = mx.array([s + [0] * (S - len(s)) for s in suffixes])
        x = self.text.embed_tokens(ids)
        for layer, layer_cache in zip(self.text.layers, cache):
            if layer.is_linear:
                branch_cache = ArraysCache(size=2)
                branch_cache[0] = mx.repeat(layer_cache[0], C, axis=0)
                branch_cache[1] = mx.repeat(layer_cache[1], C, axis=0)
                x = layer(x, mask=None, cache=branch_cache)
            else:
                x = self._branch_attention(layer, x, layer_cache.state, offset)
        x = self.text.norm(x)
        last = x[mx.arange(C), mx.array([len(s) - 1 for s in suffixes])]
        if self.lm.args.tie_word_embeddings:
            logits = self.text.embed_tokens.as_linear(last)
        else:
            logits = self.lm.lm_head(last)
        return logits

    def _score(self, context: str, prompts: list[Prompt]) -> list[dict[str, float]]:
        mx = self.mx
        sequences = [self._prompt_ids(context, p.text) for p in prompts]
        prefix_len = shared_prefix_len(sequences) if len(sequences) > 1 else len(sequences[0]) - 1
        if self._prefix_hint and self._prefix_hint < min(len(s) for s in sequences):
            prefix_len = self._prefix_hint  # two-stage: pin the prefix so both stages share one prefill
        head_len = self._stable_head_len(context, prompts[0].text, sequences[0])
        cache = self._prefill(sequences[0][:prefix_len], head_len if head_len < prefix_len else 0)
        scoring = time.perf_counter()
        n_heads = self.text.layers[self.lm.args.full_attention_interval - 1].self_attn.num_attention_heads

        # Group by (label count, suffix length) so chunks share a label set and pad little.
        order = sorted(range(len(prompts)), key=lambda i: (len(prompts[i].labels), len(sequences[i])))
        results: list[dict[str, float] | None] = [None] * len(prompts)
        start = 0
        while start < len(order):
            n_labels = len(prompts[order[start]].labels)
            chunk: list[int] = []
            for i in order[start:]:
                if len(prompts[i].labels) != n_labels or len(chunk) >= MAX_CHUNK:
                    break
                queries = (len(chunk) + 1) * (len(sequences[i]) - prefix_len)
                if chunk and n_heads * queries * (prefix_len + queries) * 4 > ATTENTION_BUDGET_BYTES:
                    break
                chunk.append(i)
            suffixes = [sequences[i][prefix_len:] for i in chunk]
            # Label ids differ per prompt (e.g. No/Yes vs A..D) but have equal counts within a chunk.
            label_matrix = mx.array([[self._label_ids[l] for l in prompts[i].labels] for i in chunk])
            picked = mx.take_along_axis(self._branch_logits(cache, suffixes, prefix_len), label_matrix, axis=1)
            probs = mx.softmax(picked.astype(mx.float32), axis=-1).tolist()
            for i, row in zip(chunk, probs):
                results[i] = dict(zip(prompts[i].options, row))
            start += len(chunk)
        if self.stats is not None:  # .tolist() above has synchronized, so this is the branches' compute
            self.stats["score_s"] += time.perf_counter() - scoring
            self.stats["score_calls"] += 1
            self.stats["branches"] += len(prompts)
            self.stats["prompt_tokens"] = max(self.stats["prompt_tokens"], max(len(s) for s in sequences))
        return results  # type: ignore[return-value]
