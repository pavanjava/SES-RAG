"""P2: SES-RAG vs baselines on the frozen eval set (same chunks, embedder, generator, budget)."""
from common import *
from ses.metrics import paired_boot

SYSTEMS = ["naive", "hybrid", "graphrag", "spectrag", "gretriever", "ses"]
COLS = ["em", "f1", "correct", "hallu", "abstain", "sp_recall", "full", "bridge_recall", "tokens", "lam2"]


def main():
    ctx = Ctx()
    qis = eval_set(ctx)
    p = best()
    kw = {"spectrag": dict(alpha=p["alpha"]), "ses": dict(mu=p["mu"], obj=p["obj"])}
    df = pd.concat([run(ctx, s, qis, **kw.get(s, {})) for s in SYSTEMS], ignore_index=True)
    save(df, "p2_rows")
    out = [f"# P2 results (budget={BUDGET}, n={len(qis)}; params {p})", ""]
    for name, g in [("ALL", df), ("bridge", df[df.type == "bridge"]), ("comparison", df[df.type == "comparison"])]:
        out += [f"## {name}", "", g.groupby("system")[COLS].mean().loc[SYSTEMS].round(3).to_markdown(), ""]
    out += ["## Paired bootstrap, SES-RAG minus baseline (mean diff, 95% CI, one-sided p that SES <= baseline)", ""]
    for scope, g in [("ALL", df), ("bridge", df[df.type == "bridge"])]:
        piv = {m: g.pivot(index="qi", columns="system", values=m) for m in ("hallu", "f1", "full", "bridge_recall")}
        for base in SYSTEMS[:-1]:
            cells = []
            for m, sign in (("hallu", -1), ("f1", 1), ("full", 1), ("bridge_recall", 1)):
                d, lo, hi, pv = paired_boot(sign * piv[m]["ses"], sign * piv[m][base])
                cells.append(f"{m}: {sign * d:+.3f} [{sign * lo if sign > 0 else sign * hi:+.3f},{sign * hi if sign > 0 else sign * lo:+.3f}] p={pv if sign > 0 else 1 - 0 if False else pv:.3f}")
            out.append(f"- {scope} vs {base}: " + "; ".join(cells))
    (RES / "p2_report.md").write_text("\n".join(out))
    print("\n".join(out))


main()
