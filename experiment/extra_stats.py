"""Extra statistics requested in review: ablation Holm, per-generator tests vs graph baselines, within-question association, external-dataset generators."""
import json
import numpy as np
from common import *
from ses.metrics import paired_boot
from analyze_paper import load, holm, MODELS  # noqa: E402 (runs analyze main on import is avoided below)

P = RES / "paper"
out = {}
d = load("hotpot", "gpt-4o-mini")
ses = d[d.system == "ses"].set_index("qi")
res = []
for v in ("pool_rel", "ses_comp", "ses_dens", "ses_lamraw"):
    g = d[d.system == v].set_index("qi")
    for m in ("full", "f1"):
        dd, lo, hi, p = paired_boot(g[m].loc[ses.index], ses[m], n=20000)
        pt = 2 * min(p, 1 - p)
        res.append([v, m, dd, lo, hi, max(pt, 1 / 20000)])
h = holm([r[5] for r in res])
out["ablation_holm"] = [dict(v=r[0], m=r[1], d=r[2], lo=r[3], hi=r[4], p2=r[5], holm=x) for r, x in zip(res, h)]
# per-generator SES - graph baselines (F1), uncorrected CIs
pg = []
for model in MODELS:
    dm = load("hotpot", model)
    s_ = dm[dm.system == "ses"].set_index("qi").f1
    for b in ("graphrag", "spectrag", "gretriever", "pool_rel", "hybrid"):
        x = dm[dm.system == b].set_index("qi").f1
        dd, lo, hi, _ = paired_boot(s_.loc[x.index], x, n=5000)
        pg.append(dict(model=model, base=b, d=dd, lo=lo, hi=hi))
out["per_generator_vs_baselines"] = pg
# within-question association (question fixed effects) of F1 with completeness, over the 8 pooled systems
fe = {}
for model in MODELS:
    dm = load("hotpot", model)
    dm = dm[dm.system.isin(["naive", "hybrid", "pool_rel", "graphrag", "spectrag", "gretriever", "ircot", "ses"])].copy()
    dm["f1c"] = dm.f1 - dm.groupby("qi").f1.transform("mean")
    dm["fc"] = dm.full - dm.groupby("qi").full.transform("mean")
    fe[model] = dict(within_slope=float((dm.f1c * dm.fc).sum() / (dm.fc ** 2).sum()),
                     raw_gap=float(dm[dm.full == 1].f1.mean() - dm[dm.full == 0].f1.mean()),
                     long_answer_rate=float((dm.answer.fillna("").str.len() >= 60).mean()))
out["within_question"] = fe
# IRCoT+SES vs IRCoT on external datasets, both generators
ext = []
for ds in ("musique", "2wiki"):
    for model in ("gpt-4o-mini", "claude-sonnet-5-5"):
        dm = load(ds, model)
        if dm is None:
            continue
        a, b = dm[dm.system == "ircot_ses"].set_index("qi"), dm[dm.system == "ircot"].set_index("qi")
        for m in ("full", "f1"):
            dd, lo, hi, _ = paired_boot(a[m].loc[b.index], b[m], n=5000)
            ext.append(dict(ds=ds, model=model, m=m, d=dd, lo=lo, hi=hi))
out["ircot_ses_vs_ircot_external"] = ext
# external-dataset generator table (F1 and gold-complete)
gen_ext = {}
for ds in ("musique", "2wiki"):
    for model in ("gpt-4o-mini", "claude-sonnet-5-5"):
        dm = load(ds, model)
        if dm is not None:
            gen_ext[f"{ds}|{model}"] = {s: dict(full=float(g.full.mean()), f1=float(g.f1.mean()), abstain=float(g.abstain.mean())) for s, g in dm.groupby("system")}
out["external_by_generator"] = gen_ext
from ses.data import load_musique, load_2wiki
out["n_paragraphs"] = dict(musique=len(load_musique().pids), wiki2=len(load_2wiki().pids))
json.dump(out, open(P / "extra_stats.json", "w"), indent=1, default=float)
print(json.dumps(out["ablation_holm"], indent=0, default=float)[:1200])
print({k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in fe.items()})
print([(x["model"], x["base"], round(100*x["d"], 1), round(100*x["lo"], 1), round(100*x["hi"], 1)) for x in pg if x["base"] in ("graphrag", "spectrag")])
print([(x["ds"], x["model"], x["m"], round(100*x["d"], 1), round(100*x["lo"], 1), round(100*x["hi"], 1)) for x in ext])
print(out["n_paragraphs"])

# table: external datasets by generator
L = ["\\begin{table}[t]", "\\centering", "\\caption{External datasets by generator: gold-complete (\\%) is generator-independent; F1 (\\%) differs by generator.}", "\\label{tab:extgen}", "\\small",
     "\\resizebox{\\columnwidth}{!}{%", "\\begin{tabular}{llccccc}", "\\toprule", "Dataset & Generator & Naive & SpectRAG-lite & IRCoT-lite & SES-RAG & Oracle \\\\", "\\midrule"]
for k, v in gen_ext.items():
    ds, model = k.split("|")
    cells = " & ".join(f"{100*v[s]['full']:.1f} / {100*v[s]['f1']:.1f}" for s in ("naive", "spectrag", "ircot", "ses", "oracle"))
    L.append(f"{'MuSiQue' if ds == 'musique' else '2Wiki'} & {model} & {cells} \\\\")
L += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
(Path(__file__).resolve().parent.parent / "paper" / "tab_ext_gen.tex").write_text("\n".join(L))
