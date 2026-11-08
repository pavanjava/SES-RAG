"""Retrieval-only sensitivity sweeps on the primary set (budget, mu), no tuning. Writes paper/sweep_*.csv"""
from common import *


def main():
    ctx = Ctx()
    fp = frozen()
    br = [i for i in ctx.d.test if ctx.d.qs[i]["type"] == "bridge"]
    qis = br[500:1361]
    ctx.ircot_prepare(qis, fp["ircot_rounds"], fp["ircot_k"])
    rows = []
    for B in (300, 450, 600, 900, 1500):
        for s in ("naive", "hybrid", "pool_rel", "graphrag", "spectrag", "gretriever", "ircot", "ses"):
            df = ctx.evaluate(s, qis, B, llm_on=False, **kwargs(s, fp))
            rows.append(dict(budget=B, system=s, full=df.full.mean(), sp_recall=df.sp_recall.mean(), tokens=df.tokens.mean()))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(RES / "paper" / "sweep_budget.csv", index=False)
    rows = []
    for mu in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0):
        for obj in ("lam", "lamraw"):
            df = ctx.evaluate("ses", qis, BUDGET, llm_on=False, mu=mu, obj=obj)
            rows.append(dict(mu=mu, obj=obj, full=df.full.mean(), sp_recall=df.sp_recall.mean(), lam2=df.lam2.mean()))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(RES / "paper" / "sweep_mu.csv", index=False)


main()
