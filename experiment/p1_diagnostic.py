"""P1: lambda2 of the context as a hallucination detector (AUROC), per system, vs relevance/length baselines."""
from sklearn.metrics import roc_auc_score
from common import *

df = pd.read_csv(RES / "p2_rows.csv")
out = ["# P1 diagnostic: AUROC for detecting hallucination (0.5 = chance)", "",
       "Score = -lam2 (low connectivity => predicted hallucination); controls: -mean_rel, tokens.", "",
       "| scope | system | n | hallu rate | AUROC(-lam2) | AUROC(-lam2_raw) | AUROC(-mean_rel) | AUROC(-ncomp==1) |", "|---|---|---|---|---|---|---|---|"]
for scope in ("bridge", "comparison"):
    for sysname, g in [("pooled", df[df.type == scope])] + list(df[df.type == scope].groupby("system")):
        if g.hallu.nunique() < 2:
            continue
        a = lambda s: roc_auc_score(g.hallu, s)
        out.append(f"| {scope} | {sysname} | {len(g)} | {g.hallu.mean():.3f} | {a(-g.lam2):.3f} | {a(-g.lam2_raw):.3f} | "
                   f"{a(-g.mean_rel):.3f} | {a((g.ncomp > 1).astype(float)):.3f} |")
(RES / "p1_report.md").write_text("\n".join(out))
print("\n".join(out))
