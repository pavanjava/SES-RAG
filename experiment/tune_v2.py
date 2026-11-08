"""v2 tuning: equal-size grids per system on dev + explore bridge questions, retrieval-only. Writes frozen params."""
import json, sys
from common import *

NOMENTION = "nomention" in sys.argv
GRID = {"spectrag": [dict(alpha=a) for a in (0.0, 0.1, 0.25, 0.5, 0.75)],
        "gretriever": [dict(n_terms=n) for n in (2, 3, 4, 5)],
        "graphrag": [dict(scale=s) for s in (0.25, 0.5, 1.0, 2.0)],
        "ses": [dict(mu=m, obj="lam") for m in (0.1, 0.25, 0.5, 1.0, 2.0)]}


def main():
    ctx = Ctx(nomention=NOMENTION)
    dev, explore, _ = splits(ctx)
    qis = dev + explore
    rows = []
    for system, grid in GRID.items():
        for kw in grid:
            df = run(ctx, system, qis, llm_on=False, **kw)
            rows.append(dict(system=system, **kw, full=df.full.mean(), sp_recall=df.sp_recall.mean(), lam2=df.lam2.mean()))
            print(rows[-1], flush=True)
    t = pd.DataFrame(rows)
    tag = "_nomention" if NOMENTION else ""
    t.to_csv(RES / f"tune_v2{tag}.csv", index=False)
    fp = {}
    for system, key in (("spectrag", "alpha"), ("gretriever", "n_terms"), ("graphrag", "scale"), ("ses", "mu")):
        g = t[t.system == system]
        fp[key] = float(g.loc[g.full.idxmax(), key]) if key != "n_terms" else int(g.loc[g.full.idxmax(), key])
    json.dump(fp, open(RES / f"frozen_params{tag}.json", "w"))
    print("frozen", fp)


main()
