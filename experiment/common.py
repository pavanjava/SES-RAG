import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd

from ses.pipeline import Ctx

RES = Path(__file__).resolve().parent / "results"
RES.mkdir(exist_ok=True)
BUDGET = 600
N_BRIDGE, N_COMP = 500, 200


def eval_set(ctx):
    test = ctx.d.test
    br = [i for i in test if ctx.d.qs[i]["type"] == "bridge"][:N_BRIDGE]
    cp = [i for i in test if ctx.d.qs[i]["type"] == "comparison"][:N_COMP]
    return br + cp


def best():
    p = RES / "best_params.json"
    return json.loads(p.read_text()) if p.exists() else {"mu": 1.0, "obj": "lam", "alpha": 0.5}


def run(ctx, system, qis, B=BUDGET, **kw):
    df = ctx.evaluate(system, qis, B, **kw)
    df["budget"] = B
    df["variant"] = system + ("" if not kw else ":" + ",".join(f"{k}={v}" for k, v in kw.items()))
    return df


def save(df, name):
    df.drop(columns=["sel"], errors="ignore").to_csv(RES / f"{name}.csv", index=False)


def splits(ctx):
    """dev bridge, explore (first 500 test bridge, seen in v1), holdout (next 500, untouched)."""
    br = [i for i in ctx.d.test if ctx.d.qs[i]["type"] == "bridge"]
    dev = [i for i in ctx.d.dev if ctx.d.qs[i]["type"] == "bridge"]
    return dev, br[:500], br[500:1000]


def frozen(tag=""):
    return json.loads((RES / f"frozen_params{tag}.json").read_text())


def kwargs(system, fp):
    return {"spectrag": dict(alpha=fp["alpha"], ), "gretriever": dict(n_terms=fp["n_terms"]),
            "graphrag": dict(scale=fp["scale"]), "ircot": dict(rounds=fp.get("ircot_rounds", 2), k=fp.get("ircot_k", 2)),
            "ircot_ses": dict(rounds=fp.get("ircot_rounds", 3), k=fp.get("ircot_k", 2), mu=fp.get("ics_mu", 0.5), beta=fp.get("ics_beta", 0.5)), "ses": dict(mu=fp["mu"], obj="lam")}.get(system, {})
