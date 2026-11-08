"""Paper runs with frozen params. Usage: paper_run.py <hotpot|musique|2wiki> [nomention] [--models m1,m2,...]
Retrieval rows are computed once per system; each generator answers them; a judge scores the answers."""
import sys
from common import *
from ses import judge as J

DATASET = next(a for a in sys.argv[1:] if a in ("hotpot", "musique", "2wiki"))
NOMENTION = "nomention" in sys.argv
MODELS = next((a.split("=")[1].split(",") for a in sys.argv if a.startswith("--models=")),
              ["gpt-4o-mini", "gpt-4o", "claude-sonnet-5-5", "claude-haiku-5-5"] if DATASET == "hotpot"
              else ["gpt-4o-mini", "claude-sonnet-5-5"])
if NOMENTION:
    MODELS = MODELS[:1]
SYSTEMS = next((a.split("=")[1].split(",") for a in sys.argv if a.startswith("--systems=")),
               ["naive", "hybrid", "pool_rel", "graphrag", "spectrag", "gretriever", "ircot", "ses", "oracle", "ircot_ses"])
EXT = next((a.split("=")[1] for a in sys.argv if a.startswith("--tag=")), "")
ABL = [("ses_dens", dict(obj="dens")), ("ses_lamraw", dict(obj="lamraw")), ("ses_comp", dict(obj="comp"))]
OUT = RES / "paper"
OUT.mkdir(exist_ok=True)


def main():
    ctx = Ctx(nomention=NOMENTION, dataset=DATASET)
    fp = frozen("_nomention" if NOMENTION else "")
    if NOMENTION:
        fp = {**frozen(), **fp}
    if DATASET == "hotpot":
        br = [i for i in ctx.d.test if ctx.d.qs[i]["type"] == "bridge"]
        qis = br[500:1361]
    else:
        qis = ctx.d.test[:500]
    ctx.ircot_prepare(qis, fp.get("ircot_rounds", 2), fp.get("ircot_k", 2))
    base = []
    for s in SYSTEMS:
        for r in ctx.evaluate(s, qis, BUDGET, llm_on=False, **kwargs(s, fp)).to_dict("records"):
            r["system"] = s
            base.append(r)
    if not NOMENTION and not EXT:
        for name, kw in ABL:
            for r in ctx.evaluate("ses", qis, BUDGET, llm_on=False, mu=fp["mu"], **kw).to_dict("records"):
                r["system"] = name
                base.append(r)
    print("retrieval rows", len(base), flush=True)
    tag = f"{DATASET}{'_nomention' if NOMENTION else ''}{'_' + EXT if EXT else ''}"
    pd.DataFrame(base).drop(columns=["sel"]).to_csv(OUT / f"retrieval_{tag}.csv", index=False)
    for mi, model in enumerate(MODELS):
        rows = [dict(r) for r in base if not (mi > 0 and r["system"].startswith("ses_") and r["system"] != "ses") and not (NOMENTION and r["system"] not in ("pool_rel", "spectrag", "ses"))]
        ctx.answer_rows(rows, shuffle=True, model=model)
        J.judge_rows(ctx, rows)
        for r in rows:
            r["sel"] = ";".join(map(str, r["sel"]))
            r["ctx_order"] = ";".join(map(str, r["ctx_order"]))
        pd.DataFrame(rows).to_csv(OUT / f"rows_{tag}_{model}.csv", index=False)
        print("done", model, flush=True)


main()
