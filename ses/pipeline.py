import random, re, time
from collections import defaultdict

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from scipy.sparse.csgraph import connected_components

from . import data, graph, llm, metrics, spectral

tok = lambda s: re.findall(r"\w+", s.lower())
minmax = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)

PROMPT = ("Answer the question using only the context. Reply with just the short answer (a few words). "
          "If the context does not contain enough information, reply exactly INSUFFICIENT.\n\n"
          "Context:\n{ctx}\n\nQuestion: {q}\nAnswer:")


QW = ("You answer multi-hop questions by searching. Question: {q}\nEvidence found so far:\n{ev}\n"
      "Write ONE short search query (under 15 words) for the information still missing to answer the question. Output only the query.")


class Ctx:
    def __init__(self, n=2000, nomention=False, dataset="hotpot"):
        self.d = {"hotpot": lambda: data.load(n), "musique": data.load_musique, "2wiki": data.load_2wiki}[dataset]()
        texts = [f"{t}: {x}" for t, x in self.d.pids]
        self.E = llm.embed(texts)
        self.QE = llm.embed([q["question"] for q in self.d.qs])
        tag = ('_nomention' if nomention else '') + ('' if dataset == 'hotpot' else '_' + dataset)
        self.tag = tag
        self.g = graph.build(self.d.pids, self.E, w_mention=0.0 if nomention else graph.W_MENTION, tag=tag)
        self.W = self.g.W
        self.bm25 = BM25Okapi([tok(t) for t in texts])
        cp = llm.CACHE / f"coords_{llm.EMBED_MODEL}{tag}.npy"
        if not cp.exists():
            np.save(cp, spectral.spectral_coords(self.W, giant=bool(tag)))
        self.coords = np.load(cp)
        self.inv = defaultdict(list)
        for i, es in enumerate(self.g.ents):
            for e in es:
                self.inv[e].append(i)
        self.ntoks = np.array(self.d.ntoks)
        self._q = {}
        self._ircot = {}

    def q(self, qi):
        if qi in self._q:
            return self._q[qi]
        cos = self.E @ self.QE[qi]
        bm = self.bm25.get_scores(tok(self.d.qs[qi]["question"]))
        rel = 0.7 * np.clip((cos - cos.mean()) / (cos.max() - cos.mean()), 0, 1) + 0.3 * bm / (bm.max() + 1e-9)
        rrf = defaultdict(float)
        for r in (np.argsort(-cos)[:100], np.argsort(-bm)[:100]):
            for rank, p in enumerate(r):
                rrf[int(p)] += 1 / (60 + rank)
        rrf_order = sorted(rrf, key=lambda p: -rrf[p])
        seeds = rrf_order[:10]
        pool = dict.fromkeys(seeds)
        frontier = seeds
        for hop, width in enumerate((8, 5)):
            nxt = []
            for p in frontier:
                ids, w = self.g.nbrs[p]
                for j in ids[np.argsort(-w)[:width]]:
                    if int(j) not in pool:
                        pool[int(j)] = None
                        nxt.append(int(j))
            frontier = nxt
        pool = np.array(sorted(pool, key=lambda p: (p not in seeds, -rel[p]))[:100])
        A = self.W[pool][:, pool].toarray()
        out = dict(cos=cos, rel=rel, rrf=rrf_order, seeds=seeds, pool=pool, A=A, rn=minmax(rel[pool]))
        self._q[qi] = out
        return out

    def fill(self, order, budget):
        sel, used = [], 0
        for p in order:
            p = int(p)
            if p not in sel and used + self.ntoks[p] <= budget:
                sel.append(p)
                used += self.ntoks[p]
        return sel

    # ---- systems ----
    def naive(self, qi, B):
        return self.fill(np.argsort(-self.q(qi)["cos"])[:50], B)

    def hybrid(self, qi, B):
        return self.fill(self.q(qi)["rrf"], B)

    def spectrag(self, qi, B, alpha=0.5):
        f = self.q(qi)
        pool, A = f["pool"], f["A"]
        seed_idx = [i for i, p in enumerate(pool) if p in f["rrf"][:5]]
        spec = (self.coords[pool] @ self.coords[pool[seed_idx]].T).max(1).clip(0)
        P = A / np.maximum(A.sum(1, keepdims=True), 1e-9)
        s0 = np.zeros(len(pool)); s0[seed_idx] = 1 / len(seed_idx)
        x = s0.copy()
        for _ in range(30):
            x = 0.15 * s0 + 0.85 * P.T @ x
        prox = (minmax(spec) + x / (x.max() + 1e-9)) / 2
        return self.fill(pool[np.argsort(-((1 - alpha) * f["rn"] + alpha * prox))], B)

    def graphrag(self, qi, B, scale=1.0):
        f = self.q(qi)
        bonus = {}
        for e in graph.spans(self.d.qs[qi]["question"]):
            for j in self.g.title2pids.get(e, ()):
                bonus[j] = 1.0
            ids = sorted(self.inv.get(e, ()), key=lambda j: -f["rel"][j])[:40]
            for j in ids:
                bonus[j] = max(bonus.get(j, 0), 0.5)
        seeds = sorted(bonus, key=lambda j: -f["rel"][j])[:3] or f["rrf"][:3]
        for s in seeds:
            for e in self.g.ents[s]:
                for j in self.g.title2pids.get(e, ()):
                    bonus[j] = max(bonus.get(j, 0), 0.25)
        for j in f["rrf"][:10]:
            bonus.setdefault(j, 0.0)
        return self.fill(sorted(bonus, key=lambda j: -(f["rel"][j] + scale * bonus[j])), B)

    def gretriever(self, qi, B, n_terms=3):
        import networkx as nx
        f = self.q(qi)
        pool, A = f["pool"], f["A"]
        G = nx.Graph()
        G.add_nodes_from(range(len(pool)))
        for a, b in zip(*np.nonzero(np.triu(A))):
            G.add_edge(int(a), int(b), cost=1 / A[a, b])
        terms = list(np.argsort(-f["rn"])[:n_terms])
        tree = set(terms)
        for comp in nx.connected_components(G):
            t = [x for x in terms if x in comp]
            if len(t) > 1:
                tree |= set(nx.algorithms.approximation.steiner_tree(G.subgraph(comp), t, weight="cost").nodes)
        order = sorted(range(len(pool)), key=lambda i: (i not in tree, -f["rn"][i]))
        return self.fill(pool[order], B)

    def ses(self, qi, B, mu=1.0, obj="lam"):
        f = self.q(qi)
        pool, A, r = f["pool"], f["A"], f["rn"]
        tk = self.ntoks[pool]
        score = {"lam": lambda M: spectral.lambda2(M),
                 "lamraw": lambda M: spectral.lambda2(M, normalized=False),
                 "comp": lambda M: 1 - connected_components(M, directed=False)[0],
                 "dens": lambda M: M.sum() / max(len(M) * (len(M) - 1), 1)}[obj]
        S, used, cur = [], 0, 0.0
        while True:
            best, best_s, best_v = None, -1e9, 0
            for j in range(len(pool)):
                if j in S or used + tk[j] > B:
                    continue
                v = score(A[np.ix_(S + [j], S + [j])]) if (len(S) and mu) else 0.0
                s = r[j] + mu * (v - cur)
                if s > best_s:
                    best, best_s, best_v = j, s, v
            if best is None:
                return [int(pool[j]) for j in S]
            S.append(best); used += tk[best]; cur = best_v

    def oracle(self, qi, B):
        f = self.q(qi)
        return self.fill(sorted(self.d.qs[qi]["gold"]) + list(np.argsort(-f["rel"])[:50]), B)

    def hybrid_order(self, cos, bm):
        rrf = defaultdict(float)
        for r in (np.argsort(-cos)[:100], np.argsort(-bm)[:100]):
            for rank, p in enumerate(r):
                rrf[int(p)] += 1 / (60 + rank)
        return sorted(rrf, key=lambda p: -rrf[p])

    def ircot_prepare(self, qis, rounds=2, k=2, writer="gpt-4o-mini"):
        key = (rounds, k)
        done = self._ircot.setdefault(key, {})
        todo = [qi for qi in qis if qi not in done]
        if not todo:
            return
        got = {qi: list(self.q(qi)["rrf"][:k]) for qi in todo}
        for _ in range(rounds):
            prompts = [QW.format(q=self.d.qs[qi]["question"], ev="\n".join(
                f"- {self.d.pids[p][0]}: {self.d.pids[p][1][:400]}" for p in got[qi])) for qi in todo]
            newq = llm.generate_many(prompts, 16, writer, 48)
            V = llm.embed([n or self.d.qs[qi]["question"] for n, qi in zip(newq, todo)])
            for qi, n, v in zip(todo, newq, V):
                order = self.hybrid_order(self.E @ v, self.bm25.get_scores(tok(n or self.d.qs[qi]["question"])))
                got[qi] += [p for p in order if p not in got[qi]][:k]
        done.update(got)

    def ircot(self, qi, B, rounds=2, k=2):
        return self.fill(self._ircot[(rounds, k)][qi], B)

    def ircot_ses(self, qi, B, rounds=3, k=2, mu=0.25, beta=0.5):
        f = self.q(qi)
        hits = self._ircot[(rounds, k)][qi]
        pool = np.array(list(dict.fromkeys(list(hits) + [int(p) for p in f["pool"]])))
        A = self.W[pool][:, pool].toarray()
        r = minmax(f["rel"][pool]) + beta * np.isin(pool, hits)
        tk = self.ntoks[pool]
        S, used, cur = [], 0, 0.0
        while True:
            best, best_s, best_v = None, -1e9, 0
            for j in range(len(pool)):
                if j in S or used + tk[j] > B:
                    continue
                v = spectral.lambda2(A[np.ix_(S + [j], S + [j])]) if (len(S) and mu) else 0.0
                sc = r[j] + mu * (v - cur)
                if sc > best_s:
                    best, best_s, best_v = j, sc, v
            if best is None:
                return [int(pool[j]) for j in S]
            S.append(best); used += tk[best]; cur = best_v

    def pool_rel(self, qi, B):
        return self.ses(qi, B, mu=0.0)

    def select(self, system, qi, B, **kw):
        return getattr(self, system)(qi, B, **kw)

    # ---- evaluation ----
    def context_stats(self, qi, sel):
        f = self.q(qi)
        M = self.W[sel][:, sel].toarray()
        return dict(lam2=spectral.lambda2(M), lam2_raw=spectral.lambda2(M, False),
                    ncomp=connected_components(M, directed=False)[0], mean_rel=float(f["rel"][sel].mean()) if sel else 0.0,
                    tokens=int(self.ntoks[sel].sum()))

    def retrieval_row(self, system, qi, B, **kw):
        q = self.d.qs[qi]
        t0 = time.perf_counter()
        sel = self.select(system, qi, B, **kw)
        ms = 1000 * (time.perf_counter() - t0)
        gold = sorted(q["gold"], key=lambda p: self.q(qi)["rel"][p])
        row = dict(qi=qi, type=q["type"], system=system, sel=sel,
                   sp_recall=len(q["gold"] & set(sel)) / len(q["gold"]), full=float(q["gold"] <= set(sel)),
                   bridge_recall=float(gold[0] in sel), ms=ms)
        return {**row, **self.context_stats(qi, sel)}

    def evaluate(self, system, qis, B, llm_on=True, workers=16, shuffle=False, model=None, **kw):
        if system in ("ircot", "ircot_ses"):
            self.ircot_prepare(qis, kw.get("rounds", 2 if system == "ircot" else 3), kw.get("k", 2))
        rows = [self.retrieval_row(system, qi, B, **kw) for qi in qis]
        if llm_on:
            self.answer_rows(rows, shuffle, model, workers)
        return pd.DataFrame(rows)

    def answer_rows(self, rows, shuffle=False, model=None, workers=16):
        def ordered(r):
            sel = list(r["sel"])
            if shuffle:
                random.Random(r["qi"]).shuffle(sel)
            return sel
        ctxs = {id(r): ordered(r) for r in rows}
        prompts = [PROMPT.format(q=self.d.qs[r["qi"]]["question"], ctx="\n".join(
            f"[{i + 1}] {self.d.pids[p][0]}: {self.d.pids[p][1]}" for i, p in enumerate(ctxs[id(r)]))) for r in rows]
        for r, a in zip(rows, llm.generate_many(prompts, workers, model)):
            gold = self.d.qs[r["qi"]]["answer"]
            r.update(answer=a, em=metrics.em(a, gold), f1=metrics.f1(a, gold), abstain=float(metrics.abstained(a)))
            r["correct"] = float(r["f1"] >= 0.5)
            r["hallu"] = float(not r["abstain"] and not r["correct"])
            text = metrics.norm(" ".join(self.d.pids[p][0] + " " + self.d.pids[p][1] for p in r["sel"]))
            r["ungrounded"] = float(not r["abstain"] and metrics.norm(a) not in text)
            r["model"] = model or llm.GEN_MODEL
            r["ctx_order"] = ctxs[id(r)]
        return rows
