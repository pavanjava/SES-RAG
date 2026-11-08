"""Tune the iterative-retrieval baseline on dev + explore bridge questions (retrieval only), add to frozen params."""
import json
from common import *


def main():
    ctx = Ctx()
    dev, explore, _ = splits(ctx)
    qis = dev + explore
    rows = []
    for rounds in (2, 3):
        for k in (2, 3):
            df = run(ctx, "ircot", qis, llm_on=False, rounds=rounds, k=k)
            rows.append(dict(rounds=rounds, k=k, full=df.full.mean(), sp_recall=df.sp_recall.mean()))
            print(rows[-1], flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(RES / "tune_ircot.csv", index=False)
    b = t.loc[t.full.idxmax()]
    fp = frozen()
    fp["ircot_rounds"], fp["ircot_k"] = int(b.rounds), int(b.k)
    json.dump(fp, open(RES / "frozen_params.json", "w"))
    print("frozen", fp)


main()
