"""Confirmatory run on bridge holdout with FROZEN params. Usage: confirm.py [musique] [nomention]"""
import sys
from common import *

DATASET = "musique" if "musique" in sys.argv else "hotpot"
NOMENTION = "nomention" in sys.argv
SYSTEMS = ["naive", "hybrid", "pool_rel", "graphrag", "spectrag", "gretriever", "ses"]
LLM_ONLY = {"pool_rel", "spectrag", "ses"} if NOMENTION else set(SYSTEMS)
ABLATIONS = [("ses", dict(mu=None, obj="dens"), "ses_dens"), ("ses", dict(mu=None, obj="lamraw"), "ses_lamraw"),
             ("ses", dict(mu=None, obj="comp"), "ses_comp")]


def main():
    ctx = Ctx(nomention=NOMENTION, dataset=DATASET)
    fp = frozen("_nomention" if NOMENTION else "")
    if DATASET == "hotpot":
        qis = splits(ctx)[2]
    else:
        qis = ctx.d.test[:500]
    tag = f"{DATASET}{'_nomention' if NOMENTION else ''}"
    parts = []
    for s in SYSTEMS:
        df = run(ctx, s, qis, shuffle=True, llm_on=s in LLM_ONLY, **kwargs(s, fp))
        df["system"] = s
        parts.append(df)
        save(pd.concat(parts, ignore_index=True), f"confirm_{tag}_rows")
        print("done", s, flush=True)
    if not NOMENTION:
        for s, kw, name in ABLATIONS:
            kw = {**kw, "mu": fp["mu"]}
            df = run(ctx, s, qis, shuffle=True, **kw)
            df["system"] = name
            parts.append(df)
            save(pd.concat(parts, ignore_index=True), f"confirm_{tag}_rows")
            print("done", name, flush=True)


main()
