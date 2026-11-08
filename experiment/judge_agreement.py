"""Agreement between the primary judge (claude-haiku-5-5) and a second judge (gpt-4o) on a 10% random subsample."""
import json
import numpy as np
from common import *
from ses import judge as J

ctx = Ctx()
out = {}
for model in ("gpt-4o-mini", "gpt-4o", "claude-sonnet-5-5", "claude-haiku-5-5"):
    f = RES / "paper" / f"rows_hotpot_{model}.csv"
    if not f.exists():
        continue
    df = pd.read_csv(f)
    df = df[df.abstain == 0]
    sub = df.sample(frac=0.10, random_state=0)
    rows = []
    for r in sub.to_dict("records"):
        r["ctx_order"] = [int(x) for x in str(r["ctx_order"]).split(";")]
        rows.append(r)
    two = J.judge_rows(ctx, [dict(r) for r in rows], judge="gpt-4o")
    a_c, b_c = np.array([r["judge_correct"] for r in rows]), np.array([r["judge_correct"] for r in two])
    a_f, b_f = np.array([r["judge_supported"] for r in rows]), np.array([r["judge_supported"] for r in two])

    def kappa(a, b):
        po = (a == b).mean(); pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
        return float((po - pe) / (1 - pe)) if pe < 1 else 1.0
    out[model] = dict(n=len(rows), correct_agree=float((a_c == b_c).mean()), correct_kappa=kappa(a_c, b_c),
                      faith_agree=float((a_f == b_f).mean()), faith_kappa=kappa(a_f, b_f),
                      f1_vs_judge_agree=float(((sub.f1 >= 0.5).values == (a_c == 1)).mean()))
    print(model, out[model], flush=True)
json.dump(out, open(RES / "paper" / "judge_agreement.json", "w"), indent=1)
