"""Tune SES mu/objective and SpectRAG alpha on the DEV split, retrieval metrics only (no LLM, no test data)."""
import json
from common import *


def main():
    ctx = Ctx()
    qis = [i for i in ctx.d.dev if ctx.d.qs[i]["type"] == "bridge"]
    rows = []
    cfgs = [("spectrag", dict(alpha=a)) for a in (0.25, 0.5, 0.75)]
    cfgs += [("ses", dict(mu=m, obj=o)) for o in ("lam", "lamraw") for m in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)]
    for system, kw in cfgs:
        df = run(ctx, system, qis, llm_on=False, **kw)
        rows.append(dict(system=system, **kw, full=df.full.mean(), sp_recall=df.sp_recall.mean(),
                         bridge_recall=df.bridge_recall.mean(), lam2=df.lam2.mean(), tokens=df.tokens.mean()))
        print(rows[-1], flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(RES / "tune_dev.csv", index=False)
    s, b = t[t.system == "ses"], t[t.system == "spectrag"]
    sb, bb = s.loc[s.full.idxmax()], b.loc[b.full.idxmax()]
    json.dump(dict(mu=float(sb.mu), obj=sb.obj, alpha=float(bb.alpha)), open(RES / "best_params.json", "w"))
    print("best", sb.to_dict(), bb.to_dict())


main()
