"""Report for a confirm_*_rows.csv. Usage: analyze.py <tag>   e.g. hotpot, musique, hotpot_nomention"""
import sys
import numpy as np
from common import *
from ses.metrics import paired_boot

tag = sys.argv[1]
df = pd.read_csv(RES / f"confirm_{tag}_rows.csv")
has_llm = "f1" in df and df.f1.notna().any()
MAIN = ["naive", "hybrid", "pool_rel", "graphrag", "spectrag", "gretriever"]
cols = ["full", "sp_recall", "lam2", "tokens"] + (["f1", "em", "hallu", "ungrounded", "abstain"] if has_llm else [])
tab = df.groupby("system")[cols].mean().round(3)
if has_llm:
    ans = df[df.abstain == 0].groupby("system").correct.mean().rename("acc_answered").round(3)
    tab = tab.join(ans)
out = [f"# Confirmatory results: {tag} (bridge only, n={df.qi.nunique()}, budget {BUDGET}, frozen params)", "", tab.to_markdown(), ""]

# paired tests SES vs each baseline on primary metrics, Holm corrected
tests = []
for base in MAIN:
    for m in ("full",) + (("f1",) if has_llm else ()):
        a = df[df.system == "ses"].set_index("qi")[m]; b = df[df.system == base].set_index("qi")[m]
        if a.isna().any() or b.isna().any():
            continue
        d, lo, hi, p = paired_boot(a.loc[b.index], b)
        tests.append((base, m, d, lo, hi, p))
order = np.argsort([t[5] for t in tests])
holm = {}
run_max = 0.0
for rank, i in enumerate(order):
    run_max = max(run_max, min(1.0, tests[i][5] * (len(tests) - rank)))
    holm[i] = run_max
out += ["## SES-RAG minus baseline (paired bootstrap, one-sided; Holm over all rows below)", "",
        "| baseline | metric | diff | 95% CI | p | Holm p |", "|---|---|---|---|---|---|"]
for i, (base, m, d, lo, hi, p) in enumerate(tests):
    out.append(f"| {base} | {m} | {d:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {p:.4f} | {holm[i]:.4f} |")
passed = all(holm[i] < 0.05 and tests[i][3] > 0 for i in range(len(tests)))
out += ["", f"**Frozen pass criterion (beat every baseline incl. mu=0 control on complete-retrieval and F1, Holm p<0.05): {'PASSED' if passed else 'NOT PASSED'}**", ""]

if has_llm:
    d2 = df.dropna(subset=["hallu"])
    out += ["## Bridge-failure link: outcome by whether the context contains both gold paragraphs", "",
            d2.groupby("full")[["correct", "hallu", "ungrounded", "abstain", "f1"]].agg(["mean"]).round(3).to_markdown(), "",
            "n per group: " + str(d2.groupby("full").size().to_dict()), ""]
abl = df[df.system.str.startswith("ses")]
if abl.system.nunique() > 1:
    out += ["## Ablations vs full SES-RAG (bridge, paired bootstrap, two-sided CI)", ""]
    ref = df[df.system == "ses"].set_index("qi")
    for v in sorted(set(abl.system) - {"ses"}):
        b = df[df.system == v].set_index("qi")
        for m in ("full",) + (("f1",) if has_llm and b.f1.notna().any() else ()):
            d, lo, hi, _ = paired_boot(b[m].loc[ref.index], ref[m])
            out.append(f"- {v} - ses, {m}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
(RES / f"confirm_{tag}_report.md").write_text("\n".join(out))
print("\n".join(out))
