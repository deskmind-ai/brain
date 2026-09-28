"""PyTorch backend for the zero-training logit baseline (CUDA on a CUDA machine; MPS works but see mlx_logits).

The shared prefix is prefilled once and its cache is replicated across batches of question suffixes.
"""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path
from typing import Any

from deskmind_brain.eval.logits_base import (  # noqa: F401  (re-exported for callers and tests)
    FINALISTS_PER_CHUNK,
    LETTERS,
    ROUND_SIZE,
    LogitsPredictorBase,
    Prompt,
    question_block,
    render_state,
    shared_prefix_len,
)

# Laptop guardrails: unified memory is shared with the OS, and exhausting it froze and rebooted a 48GB Mac
# (Qwen3.5-4B fp32 + ~9k-token prompts). Cap the MPS allocator so it raises OOM instead, and keep prompts short.
MPS_MEMORY_FRACTION = "0.6"
MPS_MAX_TOKENS, MPS_MAX_BATCH = 4096, 4


class LocalLogitsPredictor(LogitsPredictorBase):
    def __init__(
        self,
        model_id: str,
        adapter: str | None = None,
        device: str | None = None,
        dtype: str | None = None,
        share_prefix: bool = True,
        max_batch: int | None = None,
        max_tokens: int | None = None,
    ):
        if sys.platform == "darwin":
            os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", MPS_MEMORY_FRACTION)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.name = f"local:{model_id}" + (f"+{Path(adapter).name}" if adapter else "")
        path = model_id
        if not model_id.startswith(("/", ".")):
            from modelscope import snapshot_download

            path = snapshot_download(model_id)
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        # bf16 on MPS drifts by up to ~0.06 in probability between shared and unshared prefixes (linear-attention
        # layers); fp32 agrees to 1e-5, so MPS defaults to fp32.
        on_mps = self.device == "mps"
        dtype = dtype or ("float32" if on_mps else "bfloat16")
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        # Load straight onto the device so weights never exist twice in memory.
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=getattr(torch, dtype), device_map=self.device)
        if adapter:
            from peft import PeftModel

            # merging keeps inference identical to the base path (no PEFT wrapper in the forward)
            self.model = PeftModel.from_pretrained(self.model, adapter).merge_and_unload()
        self.model = self.model.eval()
        self.share_prefix = share_prefix
        self.max_batch = max_batch or (MPS_MAX_BATCH if on_mps else 16)
        self.max_tokens = max_tokens or (MPS_MAX_TOKENS if on_mps else 32768)
        self._init_labels(model_id, adapter)

    def _release_memory(self) -> None:
        if self.device == "mps":
            self.torch.mps.empty_cache()

    def _final_hidden(self, sequences: list[list[int]]):
        """Like `_final_hidden_batched`, halving the batch on CUDA/MPS out-of-memory until batch size 1."""
        torch = self.torch
        batch = self.max_batch
        while True:
            try:
                return self._final_hidden_batched(sequences, batch)
            except torch.OutOfMemoryError:
                if batch == 1:
                    raise
                batch = max(1, batch // 2)
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                elif self.device == "mps":
                    torch.mps.empty_cache()

    def _final_hidden_batched(self, sequences: list[list[int]], max_batch: int):
        """Hidden state at the last token of every sequence, sharing the common prefix when enabled."""
        torch = self.torch
        backbone = self.model.model
        prefix_len = shared_prefix_len(sequences) if self.share_prefix and len(sequences) > 1 else 0
        out = []
        with torch.inference_mode():
            cache = None
            if prefix_len > 0:
                prefix = torch.tensor([sequences[0][:prefix_len]], device=self.device)
                cache = backbone(input_ids=prefix, use_cache=True).past_key_values
            for start in range(0, len(sequences), max_batch):
                batch = [s[prefix_len:] for s in sequences[start : start + max_batch]]
                width = max(len(s) for s in batch)
                pad = self.tokenizer.pad_token_id or 0
                ids = torch.tensor([s + [pad] * (width - len(s)) for s in batch], device=self.device)
                mask = torch.tensor([[1] * len(s) + [0] * (width - len(s)) for s in batch], device=self.device)
                kwargs: dict[str, Any] = {}
                if cache is not None:
                    batch_cache = copy.deepcopy(cache)
                    batch_cache.reorder_cache(torch.zeros(len(batch), dtype=torch.long, device=self.device))
                    mask = torch.cat([torch.ones(len(batch), prefix_len, dtype=mask.dtype, device=self.device), mask], dim=1)
                    kwargs = {"past_key_values": batch_cache, "use_cache": True}
                hidden = backbone(input_ids=ids, attention_mask=mask, **kwargs).last_hidden_state
                last = torch.tensor([len(s) - 1 for s in batch], device=self.device)
                out.append(hidden[torch.arange(len(batch), device=self.device), last])
        return torch.cat(out)

    def _score(self, context: str, prompts: list[Prompt]) -> list[dict[str, float]]:
        torch = self.torch
        sequences = [self._prompt_ids(context, p.text) for p in prompts]
        if max(len(s) for s in sequences) > self.max_tokens:
            raise ValueError(f"prompt exceeds {self.max_tokens} tokens")
        hidden = self._final_hidden(sequences)
        logits = self.model.get_output_embeddings()(hidden).float()
        results = []
        for row, p in zip(logits, prompts):
            ids = torch.tensor([self._label_ids[label] for label in p.labels], device=row.device)
            probs = torch.softmax(row[ids], dim=-1).tolist()
            results.append(dict(zip(p.options, probs)))
        return results
