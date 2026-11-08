"""Design the ircot_ses extension on dev + explore only (retrieval only)."""
import json
from common import *


def main():
    ctx = Ctx()
    dev, explore, _ = splits(ctx)
    qis = dev + explore
    fp = frozen()
    ctx.ircot_prepare(qis, fp["ircot_rounds"], fp["ircot_k"])
    rows = []
    for mu in (0.0, 0.25, 0.5):
        for beta in (0.0, 0.5, 1.0, 2.0):
            df = run(ctx, "ircot_ses", qis, llm_on=False, rounds=fp["ircot_rounds"], k=fp["ircot_k"], mu=mu, beta=beta)
            rows.append(dict(mu=mu, beta=beta, full=df.full.mean(), sp_recall=df.sp_recall.mean()))
            print(rows[-1], flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(RES / "tune_ircot_ses.csv", index=False)
    b = t.loc[t.full.idxmax()]
    fp["ics_mu"], fp["ics_beta"] = float(b.mu), float(b.beta)
    json.dump(fp, open(RES / "frozen_params.json", "w"))
    print("frozen", fp)


main()
