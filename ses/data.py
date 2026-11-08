import pickle, random
from dataclasses import dataclass

import tiktoken
from datasets import load_dataset

from .llm import CACHE

_enc = tiktoken.get_encoding("cl100k_base")
ntok = lambda s: len(_enc.encode(s))


@dataclass
class Data:
    qs: list        # dicts: id, question, answer, type, gold (set of pids)
    pids: list      # corpus index -> (title, text)
    ntoks: list     # token count per paragraph
    dev: list       # indices into qs
    test: list


def load(n=2000, n_dev=300, seed=42):
    path = CACHE / f"data_{n}_{seed}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    idx = random.Random(seed).sample(range(len(ds)), n)
    key2pid, corpus, qs = {}, [], []
    for i in idx:
        r = ds[i]
        gold_titles = set(r["supporting_facts"]["title"])
        gold = set()
        for title, sents in zip(r["context"]["title"], r["context"]["sentences"]):
            text = "".join(sents).strip()
            pid = key2pid.setdefault((title, text), len(key2pid))
            if pid == len(corpus):
                corpus.append((title, text))
            if title in gold_titles:
                gold.add(pid)
        qs.append(dict(id=r["id"], question=r["question"], answer=r["answer"], type=r["type"], gold=gold))
    order = list(range(n))
    random.Random(seed).shuffle(order)
    d = Data(qs, corpus, [ntok(f"{t}: {x}") for t, x in corpus], sorted(order[:n_dev]), sorted(order[n_dev:]))
    path.write_bytes(pickle.dumps(d))
    return d


if __name__ == "__main__":
    d = load()
    from collections import Counter
    print(len(d.qs), "questions", len(d.pids), "paragraphs", Counter(q["type"] for q in d.qs))
    print("avg tokens", sum(d.ntoks) / len(d.ntoks), "dev", len(d.dev), "test", len(d.test))


def load_musique(n=1000, n_dev=200, seed=42):
    path = CACHE / f"data_musique_{n}_{seed}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    ds = load_dataset("dgslibisey/MuSiQue", split="validation")
    ok = [i for i in range(len(ds)) if ds[i]["answerable"]]
    idx = random.Random(seed).sample(ok, n)
    key2pid, corpus, qs = {}, [], []
    for i in idx:
        r = ds[i]
        gold = set()
        for p in r["paragraphs"]:
            key = (p["title"], p["paragraph_text"].strip())
            pid = key2pid.setdefault(key, len(key2pid))
            if pid == len(corpus):
                corpus.append(key)
            if p["is_supporting"]:
                gold.add(pid)
        qs.append(dict(id=r["id"], question=r["question"], answer=r["answer"], type="bridge", gold=gold))
    order = list(range(n))
    random.Random(seed).shuffle(order)
    d = Data(qs, corpus, [ntok(f"{t}: {x}") for t, x in corpus], sorted(order[:n_dev]), sorted(order[n_dev:]))
    path.write_bytes(pickle.dumps(d))
    return d


def load_2wiki(n=1000, n_dev=200, seed=42):
    path = CACHE / f"data_2wiki_{n}_{seed}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    ds = load_dataset("framolfese/2WikiMultihopQA", split="validation")
    ok = [i for i in range(len(ds)) if ds[i]["type"] in ("compositional", "inference")]
    idx = random.Random(seed).sample(ok, n)
    key2pid, corpus, qs = {}, [], []
    for i in idx:
        r = ds[i]
        gold_titles = set(r["supporting_facts"]["title"])
        gold = set()
        for title, sents in zip(r["context"]["title"], r["context"]["sentences"]):
            key = (title, "".join(sents).strip())
            pid = key2pid.setdefault(key, len(key2pid))
            if pid == len(corpus):
                corpus.append(key)
            if title in gold_titles:
                gold.add(pid)
        qs.append(dict(id=r["id"], question=r["question"], answer=r["answer"], type="bridge", gold=gold))
    order = list(range(n))
    random.Random(seed).shuffle(order)
    d = Data(qs, corpus, [ntok(f"{t}: {x}") for t, x in corpus], sorted(order[:n_dev]), sorted(order[n_dev:]))
    path.write_bytes(pickle.dumps(d))
    return d
