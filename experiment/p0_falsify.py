"""P0: does lambda2 of the Naive-RAG context predict hallucination, controlling relevance and length?"""
import numpy as np
import statsmodels.api as sm
from common import *


def main():
    ctx = Ctx()
    qis = eval_set(ctx)
    df = run(ctx, "naive", qis)
    save(df, "p0_naive_rows")
    lines = ["# P0 falsification (Naive-RAG contexts)", "",
             f"n={len(df)} budget={BUDGET}. Logit: hallu ~ lam2 + mean_rel + tokens (z-scored).", ""]
    for typ, g in [("bridge", df[df.type == "bridge"]), ("comparison", df[df.type == "comparison"]), ("all", df)]:
        for col in ("lam2", "lam2_raw"):
            X = g[[col, "mean_rel", "tokens"]].astype(float)
            X = (X - X.mean()) / (X.std() + 1e-9)
            try:
                m = sm.Logit(g["hallu"], sm.add_constant(X)).fit(disp=0)
                lines.append(f"- {typ} {col}: coef={m.params[col]:+.3f} p={m.pvalues[col]:.4f} "
                             f"(hallu rate {g.hallu.mean():.3f}, mean {col} {g[col].mean():.3f}, n={len(g)})")
            except Exception as e:
                lines.append(f"- {typ} {col}: fit failed ({type(e).__name__}: {e})")
    b = df[df.type == "bridge"]
    lines += ["", "Hallucination rate by connectivity (bridge): "
              + ", ".join(f"ncomp={k}: {v:.3f} (n={n})" for k, v, n in
                          [(k, g.hallu.mean(), len(g)) for k, g in b.groupby(b.ncomp.clip(upper=3))])]
    (RES / "p0_report.md").write_text("\n".join(lines))
    print("\n".join(lines))


main()
