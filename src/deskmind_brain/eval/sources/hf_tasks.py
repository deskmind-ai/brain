"""Public decision tasks (HuggingFace datasets mirrored on ModelScope), recast as System One requests.

Split files are downloaded once from the ModelScope mirror into data/raw/hf_tasks/ and sampled locally.
Every task is tagged:
  heldout -> never train on any split of this dataset; measures transfer to unseen task types.
  intask  -> training may use the dataset's train split; eval rows come from a split training must not touch.

References are the human labels (one-hot) unless the dataset carries annotator agreement
(civil_comments fractions, STS-B continuous similarity), which become soft distributions.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from deskmind_brain.eval.data import EvalItem, Reference, write_jsonl
from deskmind_brain.types import Question

SUITE = "hf_tasks"
MODELSCOPE = "https://www.modelscope.cn/api/v1/datasets"
MAX_FIELD_CHARS = 8000

Row = dict[str, Any]
Built = tuple[Any, dict[str, Question], dict[str, Reference]]

# ------------------------------------------------------------------------------------------ download / load


def download(repo: str, path: str, raw_dir: Path, refresh: bool = False, retries: int = 5) -> Path:
    local = raw_dir / "files" / repo / path
    if local.exists() and not refresh:
        return local
    url = f"{MODELSCOPE}/{repo}/repo?" + urllib.parse.urlencode({"Revision": "master", "FilePath": path})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "deskmind-brain-eval"}), timeout=300) as r:
                data = r.read()
            break
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == retries - 1:
                raise RuntimeError(f"{url}: {exc}") from exc
            time.sleep(2**attempt)
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(data)
    return local


def load_rows(path: Path) -> list[Row]:
    name = path.name
    if name.endswith(".parquet"):
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pylist()
    raw = path.read_bytes()
    if name.endswith(".gz"):
        raw = gzip.decompress(raw)
        name = name[: -len(".gz")]
    text = raw.decode("utf-8")
    if name.endswith(".csv"):
        return list(csv.DictReader(io.StringIO(text)))
    return [json.loads(line) for line in text.splitlines() if line.strip()]


# ------------------------------------------------------------------------------------------ helpers


def _clip(text: Any) -> Any:
    if isinstance(text, str) and len(text) > MAX_FIELD_CHARS:
        return text[:MAX_FIELD_CHARS] + " …[truncated]"
    return text


def _humanize(label: str) -> str:
    return label.replace("_", " ").strip().capitalize()


def one_hot(question: Question, value: str) -> Reference:
    options = question.options()
    if value not in options:
        raise ValueError(f"label {value!r} not in options")
    return Reference(probs={o: float(o == value) for o in options}, soft=False)


def noul_ref(p_true: Any, soft: bool = False) -> Reference:
    if isinstance(p_true, str):
        p_true = p_true.strip().lower() in ("1", "true", "yes") if not soft else float(p_true)
    p = min(max(float(p_true), 0.0), 1.0)
    return Reference(probs={"false": 1.0 - p, "true": p}, soft=soft)


@dataclass
class Task:
    name: str
    repo: str  # ModelScope dataset id
    path: str  # split file inside the repo
    role: str  # "heldout" | "intask"
    build: Callable[[Row, dict[str, Any]], Built | None]
    stratify: Callable[[Row], Any] | None = None
    # Computes shared context (e.g. the full label set) from all rows of the split.
    prepare: Callable[[list[Row]], dict[str, Any]] | None = None


def _label_set(label_col: str, text_col: str) -> Callable[[list[Row]], dict[str, Any]]:
    def prepare(rows):
        ids = {}
        for r in rows:
            ids.setdefault(r[text_col], int(r[label_col]))
        return {"labels": sorted(ids, key=lambda t: (ids[t], t))}

    return prepare


# ------------------------------------------------------------------------------------------ builders


def _intent(field: str, what: str):
    def build(row, ctx):
        q = Question(
            type="choice",
            instructions=f"Which {what} intent best matches `{field}`?",
            criteria={label: _humanize(label) for label in ctx["labels"]},
        )
        return {field: row["text"]}, {"intent": q}, {"intent": one_hot(q, row["label_text"])}

    return build


ANLI_LABELS = ["entailment", "neutral", "contradiction"]


def _anli(row, ctx):
    q = Question(
        type="choice",
        instructions="Assuming `premise` is true, how does it relate to `hypothesis`?",
        criteria={
            "entailment": "The premise guarantees the hypothesis is true",
            "neutral": "The premise neither guarantees nor rules out the hypothesis",
            "contradiction": "The premise shows the hypothesis is false",
        },
    )
    state = {"premise": row["premise"], "hypothesis": row["hypothesis"]}
    return state, {"relation": q}, {"relation": one_hot(q, ANLI_LABELS[int(row["label"])])}


EMOTION_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]


def _emotion(row, ctx):
    q = Question(
        type="choice",
        instructions="Which emotion does the author of `text` express most strongly?",
        criteria={n: _humanize(n) for n in EMOTION_LABELS},
    )
    return {"text": row["text"]}, {"emotion": q}, {"emotion": one_hot(q, EMOTION_LABELS[int(row["label"])])}


def _mmlu(row, ctx):
    letters = ["A", "B", "C", "D"]
    q = Question(
        type="choice",
        instructions="Which option correctly answers `question`?",
        criteria={letter: str(choice) for letter, choice in zip(letters, row["choices"])},
    )
    state = {"subject": _humanize(row["subject"]), "question": row["question"]}
    return state, {"answer": q}, {"answer": one_hot(q, letters[int(row["answer"])])}


def _boolq(row, ctx):
    question = row["question"].strip().rstrip("?")
    q = Question(type="noul", instructions=f"According to `passage`, is the answer to this question yes: {question}?")
    return {"passage": _clip(row["passage"])}, {"answer_is_yes": q}, {"answer_is_yes": noul_ref(row["answer"])}


def _paws(row, ctx):
    q = Question(
        type="noul",
        instructions="Do `sentence_1` and `sentence_2` mean the same thing?",
        criteria={"true": "Paraphrases: same meaning", "false": "Different meaning, even if the words overlap"},
    )
    state = {"sentence_1": row["sentence1"], "sentence_2": row["sentence2"]}
    return state, {"paraphrase": q}, {"paraphrase": noul_ref(row["label"])}


def _tweet_hate(row, ctx):
    q = Question(
        type="noul",
        instructions="Is `tweet` hate speech against immigrants or women?",
        criteria={"true": "Hateful toward immigrants or women", "false": "Not hateful toward those groups"},
    )
    return {"tweet": row["text"]}, {"hateful": q}, {"hateful": noul_ref(row["label"])}


def _tweet_irony(row, ctx):
    q = Question(type="noul", instructions="Is `tweet` ironic or sarcastic?")
    return {"tweet": row["text"]}, {"ironic": q}, {"ironic": noul_ref(row["label"])}


def _sst2(row, ctx):
    q = Question(type="noul", instructions="Does `review` express a positive opinion of the movie?")
    return {"review": row["sentence"].strip()}, {"positive": q}, {"positive": noul_ref(row["label"])}


def _toxic_chat(row, ctx):
    toxic = Question(type="noul", instructions="Is `user_input` toxic, harmful, or asking for harmful content?")
    jailbreak = Question(type="noul", instructions="Is `user_input` trying to jailbreak an AI assistant into ignoring its rules?")
    return (
        {"user_input": _clip(row["user_input"])},
        {"toxic": toxic, "jailbreak": jailbreak},
        {"toxic": noul_ref(row["toxicity"]), "jailbreak": noul_ref(row["jailbreaking"])},
    )


def _beavertails(row, ctx):
    state = {"prompt": _clip(row["prompt"]), "response": _clip(row["response"])}
    questions = {"harmful": Question(type="noul", instructions="Is `response` harmful, given `prompt`?")}
    refs = {"harmful": noul_ref(not row["is_safe"])}
    for cat, flagged in sorted(row["category"].items()):
        desc = cat.replace(",", ", ").replace("_", " ")
        questions[cat] = Question(type="noul", instructions=f"Does the exchange in `prompt` and `response` involve: {desc}?")
        refs[cat] = noul_ref(flagged)
    return state, questions, refs


CIVIL_ATTRIBUTES = {
    "toxicity": "Is `comment` rude, disrespectful, or unreasonable enough to make someone leave a discussion?",
    "insult": "Is `comment` insulting toward a person or group?",
    "threat": "Does `comment` threaten violence or harm?",
    "identity_attack": "Does `comment` attack someone because of their identity (race, religion, gender, ...)?",
    "obscene": "Is `comment` obscene or vulgar?",
}


def _civil_comments(row, ctx):
    # Attribute values are the fraction of annotators who marked the comment: a natural soft reference.
    questions = {a: Question(type="noul", instructions=text) for a, text in CIVIL_ATTRIBUTES.items()}
    refs = {a: noul_ref(row[a], soft=True) for a in CIVIL_ATTRIBUTES}
    return {"comment": _clip(row["text"])}, questions, refs


HELPSTEER_RUBRIC = {
    "helpfulness": ("How helpful is `response` for `prompt` overall?",
                    ["Not helpful at all", "Slightly helpful", "Partially helpful", "Mostly helpful", "Fully helpful"]),
    "correctness": ("How correct and complete are the facts in `response`?",
                    ["Mostly wrong or missing key facts", "Several errors or omissions", "Some errors or omissions",
                     "Minor errors or omissions", "Fully correct and complete"]),
    "coherence": ("How clear and self-consistent is `response`?",
                  ["Incoherent", "Hard to follow", "Somewhat clear with issues", "Mostly clear", "Perfectly clear and consistent"]),
    "complexity": ("How much expertise does it take to write `response`?",
                   ["Basic language anyone could write", "Simple, school-level", "Some education needed",
                    "Specialized knowledge", "Deep domain expertise"]),
    "verbosity": ("How long and detailed is `response` relative to what `prompt` asks?",
                  ["Very terse", "Short", "Moderate", "Long", "Very long and detailed"]),
}


def _helpsteer2(row, ctx):
    state = {"prompt": _clip(row["prompt"]), "response": _clip(row["response"])}
    questions, refs = {}, {}
    for attr, (instructions, levels) in HELPSTEER_RUBRIC.items():
        q = questions[attr] = Question(type="score", instructions=instructions, criteria=levels)
        refs[attr] = one_hot(q, str(int(row[attr])))
    return state, questions, refs


def _amazon_reviews(row, ctx):
    q = Question(
        type="score",
        instructions="How many stars did the author of `review` give the product?",
        criteria=["1 star: terrible", "2 stars: poor", "3 stars: average", "4 stars: good", "5 stars: excellent"],
    )
    return {"review": _clip(row["text"])}, {"stars": q}, {"stars": one_hot(q, str(int(row["label"])))}


def _stsb(row, ctx):
    q = Question(
        type="score",
        instructions="How similar in meaning are `sentence_1` and `sentence_2`?",
        criteria=[
            "Completely different topics",
            "Not equivalent, but on the same topic",
            "Not equivalent, but share some details",
            "Roughly equivalent, some important information differs",
            "Mostly equivalent, only minor details differ",
            "Completely equivalent",
        ],
    )
    # Scores are normalized to [0, 1]; spread mass between the two nearest levels so the expectation matches.
    x = min(max(float(row["score"]), 0.0), 1.0) * 5
    lo = min(int(x), 4)
    probs = {str(i): 0.0 for i in range(6)}
    probs[str(lo)], probs[str(lo + 1)] = lo + 1 - x, x - lo
    state = {"sentence_1": row["sentence1"], "sentence_2": row["sentence2"]}
    return state, {"similarity": q}, {"similarity": Reference(probs=probs, soft=True)}


TWEET_SENTIMENT = ["negative", "neutral", "positive"]


def _tweet_sentiment(row, ctx):
    q = Question(type="score", instructions="What sentiment does `tweet` express?", criteria=["Negative", "Neutral", "Positive"])
    level = TWEET_SENTIMENT.index(row["label_text"])
    return {"tweet": row["text"].strip()}, {"sentiment": q}, {"sentiment": one_hot(q, str(level))}


TASKS: list[Task] = [
    # choice
    Task("banking77", "mteb/banking77", "data/test-00000-of-00001.parquet", "intask",
         _intent("message", "banking support"), lambda r: r["label_text"], _label_set("label", "label_text")),
    Task("massive_intent", "mteb/amazon_massive_intent", "test/en.json.gz", "heldout",
         _intent("utterance", "voice assistant"), lambda r: r["label_text"], lambda rows: {"labels": sorted({r["label_text"] for r in rows})}),
    Task("anli_r3", "facebook/anli", "plain_text/test_r3-00000-of-00001.parquet", "heldout", _anli, lambda r: r["label"]),
    Task("emotion", "AI-ModelScope/emotion", "data/test-00000-of-00001.parquet", "intask", _emotion, lambda r: r["label"]),
    Task("mmlu", "cais/mmlu", "all/test-00000-of-00001.parquet", "intask", _mmlu, lambda r: r["subject"]),
    # noul
    Task("boolq", "google/boolq", "data/validation-00000-of-00001.parquet", "intask", _boolq, lambda r: r["answer"]),
    Task("paws", "google-research-datasets/paws", "labeled_final/test-00000-of-00001.parquet", "heldout", _paws, lambda r: r["label"]),
    Task("tweet_hate", "cardiffnlp/tweet_eval", "hate/test-00000-of-00001.parquet", "intask", _tweet_hate, lambda r: r["label"]),
    Task("tweet_irony", "cardiffnlp/tweet_eval", "irony/test-00000-of-00001.parquet", "heldout", _tweet_irony, lambda r: r["label"]),
    Task("sst2", "stanfordnlp/sst2", "data/validation-00000-of-00001.parquet", "intask", _sst2, lambda r: r["label"]),
    Task("toxic_chat", "lmsys/toxic-chat", "data/0124/toxic-chat_annotation_test.csv", "intask", _toxic_chat,
         lambda r: (r["toxicity"], r["jailbreaking"])),
    Task("beavertails", "PKU-Alignment/BeaverTails", "round0/30k/test.jsonl.gz", "heldout", _beavertails, lambda r: r["is_safe"]),
    Task("civil_comments", "google/civil_comments", "data/test-00000-of-00001.parquet", "heldout", _civil_comments,
         lambda r: float(r["toxicity"]) >= 0.5),
    # score
    Task("helpsteer2", "AI-ModelScope/HelpSteer2", "validation.jsonl.gz", "intask", _helpsteer2, lambda r: r["helpfulness"]),
    Task("amazon_reviews", "mteb/amazon_reviews_multi", "en/test.jsonl", "intask", _amazon_reviews, lambda r: r["label"]),
    Task("stsb", "sentence-transformers/stsb", "data/test-00000-of-00001.parquet", "heldout", _stsb, lambda r: round(float(r["score"]) * 5)),
    Task("tweet_sentiment", "mteb/tweet_sentiment_extraction", "data/test-00000-of-00001.parquet", "heldout", _tweet_sentiment,
         lambda r: r["label_text"]),
]

# ------------------------------------------------------------------------------------------ sampling


def select(task: Task, rows: list[Row], n: int, seed: int = 0) -> list[tuple[int, Row]]:
    """Shuffle, then round-robin across stratification labels so rare labels are represented."""
    rng = random.Random(f"select:{seed}:{task.name}")
    indexed = list(enumerate(rows))
    rng.shuffle(indexed)
    if task.stratify is None:
        return indexed[:n]
    buckets: dict[Any, list[tuple[int, Row]]] = defaultdict(list)
    for idx, row in indexed:
        buckets[task.stratify(row)].append((idx, row))
    order = sorted(buckets, key=str)
    rng.shuffle(order)
    picked: list[tuple[int, Row]] = []
    while len(picked) < n and any(buckets.values()):
        for key in order:
            if buckets[key] and len(picked) < n:
                picked.append(buckets[key].pop())
    return picked


def build_task(task: Task, raw_dir: Path, n: int, seed: int = 0, refresh: bool = False) -> list[EvalItem]:
    rows = load_rows(download(task.repo, task.path, raw_dir, refresh=refresh))
    ctx = task.prepare(rows) if task.prepare else {}
    items = []
    for idx, row in select(task, rows, n, seed=seed):
        built = task.build(row, ctx)
        if built is None:
            continue
        state, questions, refs = built
        items.append(
            EvalItem(
                id=f"{task.name}/{idx}",
                suite=SUITE,
                group=task.name,
                state=state,
                questions=questions,
                references=refs,
                meta={"source": f"modelscope:{task.repo}/{task.path}", "row": idx, "role": task.role},
            )
        )
    return items


def build_suite(out_dir: Path, raw_dir: Path, n: int = 100, seed: int = 0, only: list[str] | None = None, refresh: bool = False) -> dict[str, Any]:
    tasks = [t for t in TASKS if not only or t.name in only]
    all_items, summary = [], {}
    for task in tasks:
        items = build_task(task, raw_dir, n=n, seed=seed, refresh=refresh)
        all_items.extend(items)
        summary[task.name] = {"role": task.role, "items": len(items), "pairs": sum(len(it.references) for it in items)}
    write_jsonl(out_dir / "items.jsonl", (it.to_json() for it in all_items))
    result = {"suite": SUITE, "items": len(all_items), "pairs": sum(v["pairs"] for v in summary.values()), "tasks": summary}
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
