"""P3: ablations (mu=0, component count, edge density, raw lambda2), mu sweep, budget sweep."""
from common import *


def main():
    ctx = Ctx()
    qis = eval_set(ctx)
    p = best()
    mu = p["mu"]
    abl = [("ses", dict(mu=0.0, obj="lam")), ("ses", dict(mu=mu, obj="comp")), ("ses", dict(mu=mu, obj="dens")),
           ("ses", dict(mu=mu, obj="lamraw" if p["obj"] == "lam" else "lam"))]
    rows = pd.concat([run(ctx, s, qis, **kw) for s, kw in abl], ignore_index=True)
    ref = pd.read_csv(RES / "p2_rows.csv")
    ref = ref[ref.system == "ses"].assign(variant="ses:FULL")
    rows = pd.concat([ref, rows], ignore_index=True)
    save(rows, "p3_ablation_rows")
    cols = ["f1", "hallu", "abstain", "sp_recall", "full", "bridge_recall", "lam2", "tokens"]
    out = ["# P3 ablations (budget=600, LLM on)", "", rows.groupby("variant")[cols].mean().round(3).to_markdown(), ""]

    sweep = []
    for m in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0):
        g = run(ctx, "ses", qis, llm_on=False, mu=m, obj=p["obj"])
        sweep.append(dict(mu=m, **g[["sp_recall", "full", "bridge_recall", "lam2", "tokens"]].mean().round(3)))
    out += ["## mu sweep (retrieval only)", "", pd.DataFrame(sweep).to_markdown(index=False), ""]

    sub = qis[:300]
    bs = []
    for B in (300, 400, 900, 1500):
        for s, kw in [("naive", {}), ("hybrid", {}), ("graphrag", {}), ("spectrag", dict(alpha=p["alpha"])),
                      ("gretriever", {}), ("ses", dict(mu=mu, obj=p["obj"]))]:
            g = run(ctx, s, sub, B=B, llm_on=B in (400, 900), **kw)
            bs.append(dict(budget=B, system=s, **g.drop(columns=["sel"]).select_dtypes("number").drop(columns=["qi", "budget"]).mean()[
                [c for c in ["f1", "hallu", "sp_recall", "full", "bridge_recall", "lam2", "tokens"] if c in g]].round(3)))
    bsd = pd.DataFrame(bs)
    bsd.to_csv(RES / "p3_budget_sweep.csv", index=False)
    out += ["## budget sweep (first 300 eval questions; LLM metrics only at 400 and 900)", "", bsd.to_markdown(index=False)]
    (RES / "p3_report.md").write_text("\n".join(out))
    print("\n".join(out))


main()
