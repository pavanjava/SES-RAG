"""Where does SES-RAG fail? Graph adjacency of gold pairs, pool coverage, and selection losses (primary set)."""
import json
import numpy as np
from common import *


def main():
    ctx = Ctx()
    fp = frozen()
    br = [i for i in ctx.d.test if ctx.d.qs[i]["type"] == "bridge"]
    qis = br[500:1361]
    out = {}
    adj, inpool, sel_ok, seed_hit, linked = [], [], [], [], []
    miss_pool, miss_sel = 0, 0
    for qi in qis:
        gold = sorted(ctx.d.qs[qi]["gold"])
        f = ctx.q(qi)
        W = ctx.W[gold][:, gold].toarray()
        adj.append(float((W + np.eye(len(gold)) * 0 > 0).any(1).all()))
        pool = set(int(p) for p in f["pool"])
        inpool.append(float(set(gold) <= pool))
        seeds = set(f["seeds"])
        seed_hit.append(float(len(seeds & set(gold)) >= 1))
        sel = ctx.select("ses", qi, BUDGET, mu=fp["mu"], obj="lam")
        ok = set(gold) <= set(sel)
        sel_ok.append(float(ok))
        if not ok:
            if set(gold) <= pool:
                miss_sel += 1
            else:
                miss_pool += 1
    n = len(qis)
    out = dict(n=n, gold_nodes_all_have_edge_to_another_gold=float(np.mean(adj)), gold_in_pool=float(np.mean(inpool)),
               at_least_one_gold_is_seed=float(np.mean(seed_hit)), ses_complete=float(np.mean(sel_ok)),
               ses_fail_gold_not_in_pool=miss_pool / n, ses_fail_in_pool_not_selected=miss_sel / n)
    json.dump(out, open(RES / "paper" / "error_analysis.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


main()
