import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if not list((Path(__file__).resolve().parent.parent / "cache").glob("graph_*.pkl")):
    pytest.skip("caches not built", allow_module_level=True)

from ses.pipeline import Ctx


@pytest.fixture(scope="module")
def ctx():
    return Ctx()


SYSTEMS = ["naive", "hybrid", "pool_rel", "graphrag", "spectrag", "gretriever", "ses"]


@pytest.mark.parametrize("system", SYSTEMS)
def test_budget_and_uniqueness(ctx, system):
    for qi in ctx.d.dev[:15]:
        sel = ctx.select(system, qi, 600)
        assert sel and len(sel) == len(set(sel))
        assert ctx.ntoks[sel].sum() <= 600


def test_ses_mu0_equals_pool_rel(ctx):
    for qi in ctx.d.dev[:15]:
        assert ctx.select("ses", qi, 600, mu=0.0) == ctx.select("pool_rel", qi, 600)


def test_deterministic(ctx):
    qi = ctx.d.dev[0]
    assert ctx.select("ses", qi, 600, mu=0.25) == ctx.select("ses", qi, 600, mu=0.25)


def test_bigger_budget_never_smaller_context(ctx):
    qi = ctx.d.dev[1]
    assert ctx.ntoks[ctx.select("ses", qi, 1500, mu=0.25)].sum() >= ctx.ntoks[ctx.select("ses", qi, 300, mu=0.25)].sum()
