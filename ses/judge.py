from . import llm

CORRECT = ("Question: {q}\nGold answer: {g}\nPredicted answer: {a}\n"
           "Is the predicted answer correct, i.e. equivalent in meaning to the gold answer? Reply YES or NO.")
FAITHFUL = ("Context:\n{ctx}\n\nQuestion: {q}\nAnswer: {a}\n"
            "Is the answer fully supported by the context above (not relying on outside knowledge)? Reply YES or NO.")


def judge_rows(ctx, rows, judge="claude-haiku-5-5", workers=16):
    """Adds judge_correct and judge_supported to rows that have an answer; abstentions are not judged."""
    live = [r for r in rows if not r["abstain"]]
    pc = [CORRECT.format(q=ctx.d.qs[r["qi"]]["question"], g=ctx.d.qs[r["qi"]]["answer"], a=r["answer"]) for r in live]
    pf = [FAITHFUL.format(q=ctx.d.qs[r["qi"]]["question"], a=r["answer"], ctx="\n".join(
        f"[{i + 1}] {ctx.d.pids[p][0]}: {ctx.d.pids[p][1]}" for i, p in enumerate(r["ctx_order"]))) for r in live]
    yes = lambda t: float(t.strip().upper().startswith("YES"))
    for r, c, f in zip(live, llm.generate_many(pc, workers, judge, 5), llm.generate_many(pf, workers, judge, 5)):
        r["judge_correct"], r["judge_supported"] = yes(c), yes(f)
    for r in rows:
        if r["abstain"]:
            r["judge_correct"], r["judge_supported"] = 0.0, 1.0
        r["judge_hallu"] = float(not r["abstain"] and not r["judge_supported"])
        r["judge_wrong"] = float(not r["abstain"] and not r["judge_correct"])
    return rows
