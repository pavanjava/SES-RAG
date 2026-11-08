import hashlib, json, os, pickle, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()
ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
CACHE.mkdir(exist_ok=True)
OLLAMA = os.getenv("OLLAMA_URL", "http://localhost:11434")
GEN_MODEL = os.getenv("GEN_MODEL", "gemma3")
USE_OPENAI = bool(os.getenv("OPENAI_API_KEY"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small") if USE_OPENAI else os.getenv("LOCAL_EMBED_MODEL", "nomic-embed-text-v2-moe")

_lock = threading.Lock()
_h = lambda s: hashlib.sha1(s.encode()).hexdigest()


def _embed_batch(texts):
    if USE_OPENAI:
        from openai import OpenAI
        out = OpenAI().embeddings.create(model=EMBED_MODEL, input=texts)
        return [d.embedding for d in out.data]
    r = requests.post(f"{OLLAMA}/api/embed", json={"model": EMBED_MODEL, "input": texts}, timeout=600)
    r.raise_for_status()
    return r.json()["embeddings"]


_stores = {}


def embed(texts, batch=64):
    path = CACHE / f"emb_{EMBED_MODEL}.pkl"
    if EMBED_MODEL not in _stores:
        _stores[EMBED_MODEL] = pickle.loads(path.read_bytes()) if path.exists() else {}
    store = _stores[EMBED_MODEL]
    todo = [t for t in dict.fromkeys(texts) if _h(t) not in store]
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        for t, v in zip(chunk, _embed_batch(chunk)):
            store[_h(t)] = np.asarray(v, dtype=np.float32)
        if (i // batch) % 20 == 0:
            print(f"embedded {i + len(chunk)}/{len(todo)}", flush=True)
    if todo:
        with _lock:
            path.write_bytes(pickle.dumps(store))
    X = np.stack([store[_h(t)] for t in texts])
    return X / np.linalg.norm(X, axis=1, keepdims=True)


_gen_path = CACHE / "gen_all.jsonl"
_legacy_path = CACHE / f"gen_{GEN_MODEL}.jsonl"
_gen = {}
for _p in (_legacy_path, _gen_path):
    if _p.exists():
        for line in _p.read_text().splitlines():
            k, v = json.loads(line)
            _gen[k] = v

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")


def _call(model, prompt, num_predict):
    if model.startswith("gpt"):
        from openai import OpenAI
        r = OpenAI().chat.completions.create(model=model, temperature=0, max_tokens=num_predict, seed=0,
                                             messages=[{"role": "user", "content": prompt}])
        return r.choices[0].message.content
    if model.startswith("claude"):
        r = requests.post("https://api.anthropic.com/v1/messages", timeout=120,
                          headers={"x-api-key": ANTHROPIC_KEY, "anthropic-version": "2023-06-01"},
                          json={"model": model, "max_tokens": num_predict,
                                "thinking": {"type": "disabled" if "haiku" in model else "between_tools"},
                                "messages": [{"role": "user", "content": prompt}]})
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json().get("content", []) if b.get("type") == "text")
    if model.startswith("gemini"):
        r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent", timeout=120,
                          headers={"x-goog-api-key": GEMINI_KEY},
                          json={"contents": [{"parts": [{"text": prompt}]}],
                                "generationConfig": {"temperature": 0, "maxOutputTokens": num_predict,
                                                     **({"thinkingConfig": {"thinkingBudget": 0}} if "2.5-flash" in model and "lite" not in model else {})}})
        r.raise_for_status()
        c = r.json()["candidates"][0]
        return "".join(p.get("text", "") for p in c.get("content", {}).get("parts", []))
    r = requests.post(f"{OLLAMA}/api/generate", timeout=300, json={
        "model": model, "prompt": prompt, "stream": False, "keep_alive": "30m",
        "options": {"temperature": 0, "seed": 0, "num_predict": num_predict}})
    r.raise_for_status()
    return r.json()["response"]


def generate(prompt, model=None, num_predict=40):
    import time
    model = model or GEN_MODEL
    k = _h(f"{num_predict}|{prompt}") if model == GEN_MODEL and not model.startswith(("gpt", "claude", "gemini")) else _h(f"{model}|{num_predict}|{prompt}")
    if k in _gen:
        return _gen[k]
    for attempt in range(8):
        try:
            text = (_call(model, prompt, num_predict) or "").strip()
            if not text:
                raise ValueError("empty")
            break
        except Exception as e:
            if attempt == 7 and isinstance(e, ValueError):
                return ""
            if attempt == 7:
                raise RuntimeError(f"{model} failed: {type(e).__name__}") from None
            time.sleep(min(60, 2 ** attempt))
    with _lock:
        _gen[k] = text
        with open(_gen_path, "a") as f:
            f.write(json.dumps([k, text]) + "\n")
    return text


def generate_many(prompts, workers=3, model=None, num_predict=40):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda p: generate(p, model, num_predict), prompts))


if __name__ == "__main__":
    print("embed", embed(["hello world"]).shape, EMBED_MODEL)
    for m in ["gpt-4o-mini", "gpt-4o", "gemini-2.5-flash", "gemini-2.5-flash-lite", "claude-sonnet-5-5", "claude-haiku-5-5"]:
        try:
            print(m, "->", repr(generate("Reply with the single word OK.", m, 10)))
        except Exception as e:
            print(m, "FAILED", e)
