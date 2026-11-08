"""Builds paper tables (LaTeX), figures and numbers.json from experiment/results/paper/*.csv"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *
from ses.metrics import paired_boot

P = RES / "paper"
FIG = Path(__file__).resolve().parent.parent / "paper" / "figs"
TAB = Path(__file__).resolve().parent.parent / "paper"
FIG.mkdir(exist_ok=True)
PRIMARY = "gpt-4o-mini"
MODELS = ["gpt-4o-mini", "gpt-4o", "claude-sonnet-5-5", "claude-haiku-5-5"]
LABEL = {"naive": "Naive RAG", "hybrid": "Hybrid RAG", "pool_rel": "Hybrid+Expand ($\\mu{=}0$)", "graphrag": "GraphRAG-lite",
         "spectrag": "SpectRAG-lite", "gretriever": "G-Retriever-lite", "ircot": "IRCoT-lite", "ses": "SES-RAG",
         "oracle": "Gold-first oracle", "ircot_ses": "IRCoT+SES", "ses_dens": "SES, density term",
         "ses_lamraw": "SES, raw $\\lambda_2$", "ses_comp": "SES, component count"}
MAIN = ["naive", "hybrid", "pool_rel", "graphrag", "spectrag", "gretriever", "ircot"]
ORDER = MAIN + ["ses", "ircot_ses", "oracle"]
CALLS = {"ircot": 3, "ircot_ses": 3}
nums = {}


def load(ds, model, ext=True):
    parts = []
    for tag in ("", "_ext") if ext else ("",):
        f = P / f"rows_{ds}{tag}_{model}.csv"
        if f.exists():
            parts.append(pd.read_csv(f))
    return pd.concat(parts, ignore_index=True) if parts else None


def ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    m = x[np.random.default_rng(seed).integers(0, len(x), (n, len(x)))].mean(1)
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def fp(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def holm(ps):
    o = np.argsort(ps); out = np.zeros(len(ps)); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, ps[i] * (len(ps) - r))); out[i] = run
    return out


def cell(c):
    return f"{100*c[0]:.1f}\\,{{\\scriptsize[{100*c[1]:.1f},{100*c[2]:.1f}]}}"


def table(df, systems, cols, heads, cap, lab, star=True, wide=True):
    env = "table*" if wide else "table"
    out = [f"\\begin{{{env}}}[t]", "\\centering", f"\\caption{{{cap}}}", f"\\label{{{lab}}}", "\\small",
           "\\begin{tabular}{l" + "c" * (len(cols) + 1) + "}", "\\toprule", "System & " + " & ".join(heads) + " & LLM calls \\\\", "\\midrule"]
    for s in systems:
        g = df[df.system == s]
        if g.empty:
            continue
        if s == "oracle":
            out.append("\\midrule")
        name = LABEL[s]
        name = f"\\textbf{{{name}}}" if s == "ses" else name
        out.append(f"{name} & " + " & ".join(cell(ci(g[c])) if c != "ans_acc" else f"{100*g[g.abstain==0].judge_correct.mean():.1f}" for c in cols)
                   + f" & {CALLS.get(s, 0) if s != 'oracle' else '--'} \\\\")
    out += ["\\bottomrule", "\\end{tabular}", f"\\end{{{env}}}"]
    return "\n".join(out)


def tests(df, base_list, metrics_=("full", "f1")):
    rows = []
    ses = df[df.system == "ses"].set_index("qi")
    for b in base_list:
        base = df[df.system == b].set_index("qi")
        for m in metrics_:
            d, lo, hi, p = paired_boot(ses[m].loc[base.index], base[m], n=5000)
            rows.append([b, m, d, lo, hi, p])
    h = holm([r[5] for r in rows])
    return [r + [x] for r, x in zip(rows, h)]


def tex_tests(rows, cap, lab):
    out = ["\\begin{table}[t]", "\\centering", f"\\caption{{{cap}}}", f"\\label{{{lab}}}", "\\small", "\\begin{tabular}{llccc}", "\\toprule",
           "Baseline & Metric & $\\Delta$ (pts) & 95\\% CI & Holm $p$ \\\\", "\\midrule"]
    for b, m, d, lo, hi, p, hp in rows:
        out.append(f"{LABEL[b]} & {'Gold-complete' if m == 'full' else 'F1'} & {100*d:+.1f} & [{100*lo:+.1f}, {100*hi:+.1f}] & {fp(hp)} \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(out)


def main():
    df = load("hotpot", PRIMARY)
    if df is None:
        print("no primary rows"); return
    nums["n_primary"] = int(df.qi.nunique())
    cols = ["full", "f1", "judge_correct", "abstain"]
    heads = ["Gold-complete (\\%)", "F1 (\\%)", "Judged acc. (\\%)", "Abstain (\\%)"]
    (TAB / "tab_main.tex").write_text(table(df, ORDER, cols, heads,
        f"HotpotQA bridge questions ($n={nums['n_primary']}$), generator gpt-4o-mini, 600-token budget. Mean with 95\\% bootstrap CI. LLM calls = query-time LLM calls before answering.", "tab:main"))
    t = tests(df, MAIN)
    (TAB / "tab_tests.tex").write_text(tex_tests(t, "SES-RAG minus each baseline, primary set, gpt-4o-mini (paired one-sided bootstrap, Holm-corrected over the 14 comparisons).", "tab:tests"))
    nums["tests"] = [dict(base=b, metric=m, diff=d, lo=lo, hi=hi, p=p, holm=hp) for b, m, d, lo, hi, p, hp in t]
    nums["criterion_passed"] = bool(all(r[6] < 0.05 and r[3] > 0 for r in t))
    nums["beats_all_llm_free"] = bool(all(r[6] < 0.05 and r[3] > 0 for r in t if r[0] != "ircot"))
    # extension vs ircot and ses
    ext = []
    for a, b in (("ircot_ses", "ircot"), ("ircot_ses", "ses")):
        for m in ("full", "f1"):
            A, B = df[df.system == a].set_index("qi")[m], df[df.system == b].set_index("qi")[m]
            if len(A):
                d, lo, hi, p = paired_boot(A.loc[B.index], B, n=5000)
                ext.append(dict(a=a, b=b, metric=m, diff=d, lo=lo, hi=hi, p=p))
    nums["ext"] = ext
    # ablations
    ab = []
    ses = df[df.system == "ses"].set_index("qi")
    for s in ("pool_rel", "ses_comp", "ses_dens", "ses_lamraw"):
        g = df[df.system == s].set_index("qi")
        if len(g):
            r = dict(system=s, full=g.full.mean(), f1=g.f1.mean())
            for m in ("full", "f1"):
                d, lo, hi, _ = paired_boot(g[m].loc[ses.index], ses[m], n=5000)
                r[f"d_{m}"] = (d, lo, hi)
            ab.append(r)
    nums["ablation"] = ab
    lines = ["\\begin{table}[t]", "\\centering", "\\caption{Ablations of the selection objective (primary set, gpt-4o-mini). $\\Delta$ = variant minus SES-RAG, 95\\% CI.}", "\\label{tab:abl}", "\\small",
             "\\begin{tabular}{lcccc}", "\\toprule", "Variant & Gold-complete & $\\Delta$ & F1 & $\\Delta$ \\\\", "\\midrule",
             f"SES-RAG (norm.\\ $\\lambda_2$) & {100*ses.full.mean():.1f} & -- & {100*ses.f1.mean():.1f} & -- \\\\"]
    for r in ab:
        lines.append(f"{LABEL[r['system']]} & {100*r['full']:.1f} & {100*r['d_full'][0]:+.1f} [{100*r['d_full'][1]:+.1f},{100*r['d_full'][2]:+.1f}] & {100*r['f1']:.1f} & {100*r['d_f1'][0]:+.1f} [{100*r['d_f1'][1]:+.1f},{100*r['d_f1'][2]:+.1f}] \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (TAB / "tab_abl.tex").write_text("\n".join(lines))
    # generators: F1 per system + link
    rob, link = [], []
    for model in MODELS:
        d = load("hotpot", model)
        if d is None:
            continue
        r = dict(model=model)
        for s in ORDER:
            g = d[d.system == s]
            if len(g):
                r[s] = ci(g.f1)
        rob.append(r)
        dd = d[d.system.isin(MAIN + ["ses"])]
        for full, g in dd.groupby("full"):
            link.append(dict(model=model, complete=int(full), n=len(g), acc=g.judge_correct.mean(), f1=g.f1.mean(), abstain=g.abstain.mean()))
        ses_, hy = d[d.system == "ses"].set_index("qi").f1, d[d.system == "hybrid"].set_index("qi").f1
        r["ses_minus_hybrid"] = paired_boot(ses_.loc[hy.index], hy, n=5000)[:3]
        ir = d[d.system == "ircot"].set_index("qi").f1
        r["ses_minus_ircot"] = paired_boot(ses_.loc[ir.index], ir, n=5000)[:3]
    nums["robust"], nums["link"] = rob, link
    sh = ["naive", "hybrid", "pool_rel", "spectrag", "ircot", "ses", "oracle"]
    L = ["\\begin{table*}[t]", "\\centering", "\\caption{Answer F1 (\\%) by generator on the primary set (same contexts, different generators).}", "\\label{tab:gen}", "\\small",
         "\\begin{tabular}{l" + "c" * len(sh) + "}", "\\toprule", "Generator & " + " & ".join(LABEL[s] for s in sh) + " \\\\", "\\midrule"]
    for r in rob:
        L.append(r["model"] + " & " + " & ".join(f"{100*r[s][0]:.1f}" for s in sh) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    (TAB / "tab_gen.tex").write_text("\n".join(L))
    # link table
    lk = pd.DataFrame(link)
    if len(lk):
        L = ["\\begin{table}[t]", "\\centering", "\\caption{Outcome by whether the context contains all gold paragraphs (pooled over the eight non-oracle systems on the primary set).}", "\\label{tab:link}", "\\small",
             "\\begin{tabular}{llrccc}", "\\toprule", "Generator & Context & $n$ & Judged acc. (\\%) & F1 (\\%) & Abstain (\\%) \\\\", "\\midrule"]
        for _, r in lk.iterrows():
            L.append(f"{r.model} & {'complete' if r.complete else 'bridge-failed'} & {int(r.n)} & {100*r.acc:.1f} & {100*r.f1:.1f} & {100*r.abstain:.1f} \\\\")
        L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
        (TAB / "tab_link.tex").write_text("\n".join(L))
        fig, ax = plt.subplots(figsize=(3.4, 2.4))
        w = 0.35
        ms_ = list(lk.model.unique())
        for i, c in enumerate((1, 0)):
            sub = lk[lk.complete == c].set_index("model").loc[ms_]
            ax.bar(np.arange(len(ms_)) + i * w, 100 * sub.f1, w, label="complete" if c else "bridge-failed")
        ax.set_xticks(np.arange(len(ms_)) + w / 2); ax.set_xticklabels([m.replace("claude-", "").replace("-5-5", "") for m in ms_], fontsize=6, rotation=15)
        ax.set_ylabel("F1 (%)"); ax.legend(fontsize=6); fig.tight_layout(); fig.savefig(FIG / "fig_link.pdf"); plt.close(fig)
    # sweeps
    f = P / "sweep_budget.csv"
    if f.exists():
        b = pd.read_csv(f)
        fig, ax = plt.subplots(figsize=(3.4, 2.5))
        for s in ["naive", "hybrid", "pool_rel", "spectrag", "ircot", "ses"]:
            g = b[b.system == s]
            ax.plot(g.budget, 100 * g.full, marker="o", ms=3, lw=1.6 if s == "ses" else 0.9, label=LABEL[s].replace("$", "").replace("\\mu{=}0", "mu=0"))
        ax.set_xlabel("token budget"); ax.set_ylabel("gold-complete (%)"); ax.legend(fontsize=5); fig.tight_layout(); fig.savefig(FIG / "fig_budget.pdf"); plt.close(fig)
        nums["budget"] = b.pivot(index="system", columns="budget", values="full").round(3).to_dict()
    f = P / "sweep_mu.csv"
    if f.exists():
        m = pd.read_csv(f)
        fig, ax = plt.subplots(figsize=(3.4, 2.3))
        for o, lab in (("lam", "normalised $\\lambda_2$"), ("lamraw", "raw $\\lambda_2$")):
            g = m[m.obj == o]; ax.plot(g.mu, 100 * g.full, marker="o", ms=3, label=lab)
        ax.set_xscale("symlog", linthresh=0.1); ax.set_xlabel("$\\mu$"); ax.set_ylabel("gold-complete (%)"); ax.legend(fontsize=6); fig.tight_layout(); fig.savefig(FIG / "fig_mu.pdf"); plt.close(fig)
        nums["mu_sweep"] = m.round(3).to_dict("records")
    # other datasets, no-mention
    for ds in ("musique", "2wiki"):
        d = load(ds, PRIMARY)
        if d is None:
            continue
        sub = ["naive", "hybrid", "pool_rel", "graphrag", "spectrag", "gretriever", "ircot", "ses", "ircot_ses", "oracle"]
        (TAB / f"tab_{ds}.tex").write_text(table(d, sub, ["full", "f1", "judge_correct"], ["Gold-complete (\\%)", "F1 (\\%)", "Judged acc. (\\%)"],
            f"{'MuSiQue' if ds == 'musique' else '2WikiMultiHopQA'} ($n={d.qi.nunique()}$ questions, settings frozen on HotpotQA), gpt-4o-mini.", f"tab:{ds}", wide=False))
        tt = tests(d, MAIN)
        nums[f"tests_{ds}"] = [dict(base=b_, metric=m_, diff=d_, lo=lo, hi=hi, holm=hp) for b_, m_, d_, lo, hi, p, hp in tt]
    r_nm = P / "retrieval_hotpot_nomention.csv"
    r_m = P / "retrieval_hotpot.csv"
    if r_nm.exists() and r_m.exists():
        a, b = pd.read_csv(r_m), pd.read_csv(r_nm)
        rows = []
        for s in MAIN + ["ses", "ircot_ses"]:
            ga, gb = a[a.system == s], b[b.system == s]
            if len(ga) and len(gb):
                rows.append((s, ga.full.mean(), gb.full.mean()))
        nums["nomention"] = rows
        nm_llm = load("hotpot_nomention", PRIMARY, ext=False)
        f1 = nm_llm.groupby("system").f1.mean().to_dict() if nm_llm is not None else {}
        L = ["\\begin{table}[t]", "\\centering", "\\caption{Primary set with and without title-mention edges in the graph: gold-complete rate (\\%) and, where answered with gpt-4o-mini, F1 (\\%).}", "\\label{tab:nomention}", "\\small",
             "\\begin{tabular}{lccc}", "\\toprule", "System & Gold-complete, with & Gold-complete, without & F1, without \\\\", "\\midrule"]
        for s_, x, y in rows:
            L.append(f"{LABEL[s_]} & {100*x:.1f} & {100*y:.1f} & {100*f1[s_]:.1f} \\\\" if s_ in f1 else f"{LABEL[s_]} & {100*x:.1f} & {100*y:.1f} & -- \\\\")
        L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
        (TAB / "tab_nomention.tex").write_text("\n".join(L))
    # transfer tests table
    rowsT = []
    for ds, name in (("musique", "MuSiQue"), ("2wiki", "2Wiki")):
        for t_ in nums.get(f"tests_{ds}", []):
            rowsT.append((name, t_))
    if rowsT:
        L = ["\\begin{table}[t]", "\\centering", "\\caption{SES-RAG minus each baseline on the external datasets (gpt-4o-mini; $\\Delta$ in points, 95\\% CI, Holm $p$ over the 14 comparisons per dataset).}", "\\label{tab:transfer_tests}", "\\small",
             "\\begin{tabular}{llllcc}", "\\toprule", "Dataset & Baseline & Metric & $\\Delta$ & 95\\% CI & Holm $p$ \\\\", "\\midrule"]
        for name, t_ in rowsT:
            L.append(f"{name} & {LABEL[t_['base']]} & {'Gold-complete' if t_['metric'] == 'full' else 'F1'} & {100*t_['diff']:+.1f} & [{100*t_['lo']:+.1f}, {100*t_['hi']:+.1f}] & {fp(t_['holm'])} \\\\")
        L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
        (TAB / "tab_transfer_tests.tex").write_text("\n".join(L))
    # cost
    ms = df.groupby("system").ms.mean().round(2).to_dict()
    nums["ms"] = ms
    nums["judge_agreement"] = json.load(open(P / "judge_agreement.json")) if (P / "judge_agreement.json").exists() else {}
    import re
    for f in TAB.glob("tab_*.tex"):
        t = f.read_text()
        w = "\\textwidth" if "table*" in t else "\\columnwidth"
        t = re.sub(r"(\\begin\{tabular\}.*?\\end\{tabular\})", lambda m: "\\resizebox{" + w + "}{!}{%\n" + m.group(1) + "}", t, flags=re.S)
        f.write_text(t)
    json.dump(nums, open(P / "numbers.json", "w"), indent=1, default=float)
    print("analysis ok:", sorted(nums))


if __name__ == "__main__":
    main()
