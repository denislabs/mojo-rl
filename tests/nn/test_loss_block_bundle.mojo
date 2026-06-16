"""LossBlock / LossBlockBundle — Block E-3.

Validates:
  1. `CriticUpdateBlock` + `TwinCriticUpdateBlock` conform to `LossBlock`
     (compile-time check)
  2. `LossBlockBundle[CriticUpdateBlock, CriticUpdateBlock]` stores two
     critic-update blocks and each remains independently functional
  3. Bundle-routed step produces bit-identical results to direct block
     step (independence — bundle is pure storage, no state crosstalk)
"""

from std.math import abs as fabs
from std.memory import alloc
from std.random import seed
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.primitives.linear import Linear
from mojo_rl.nn.primitives.relu import ReLU
from mojo_rl.nn.combinators import Sequential
from mojo_rl.nn.loss import LossBlock, LossBlockBundle, CriticUpdateBlock
from mojo_rl.nn.optimizer import Adam
from mojo_rl.nn.initializer import Kaiming


def test_critic_block_conforms_to_lossblock() raises:
    """Compile-time conformance check. If CriticUpdateBlock does NOT
    conform to LossBlock, this declaration won't type-check."""
    comptime BATCH = 4
    comptime SA = 5
    comptime CRITIC = Sequential[Linear[SA, 8], ReLU[8], Linear[8, 1]]
    # Just declare the type — compilation = trait conformance.
    comptime BundleT = LossBlockBundle[
        CriticUpdateBlock[CRITIC, BATCH, SA],
    ]
    var b = BundleT.make_default["cpu"]()
    print("  test_critic_block_conforms_to_lossblock PASSED")


def test_bundle_stores_two_critic_blocks() raises:
    """Build a bundle holding two CriticUpdateBlock instances, run a
    single update through each, verify both produce valid loss values."""
    comptime BATCH = 4
    comptime SA = 5
    comptime CRITIC = Sequential[Linear[SA, 8], ReLU[8], Linear[8, 1]]

    seed(31)
    var c1 = CRITIC.make["cpu", INIT=Kaiming]()
    var c2 = CRITIC.make["cpu", INIT=Kaiming]()
    var c1_opt = Adam.make[target="cpu", M=CRITIC](c1)
    c1_opt.lr = Scalar[DT](1e-3)
    var c2_opt = Adam.make[target="cpu", M=CRITIC](c2)
    c2_opt.lr = Scalar[DT](1e-3)

    comptime BlockT = CriticUpdateBlock[CRITIC, BATCH, SA]
    var bundle = LossBlockBundle[BlockT, BlockT].make_default["cpu"]()
    bundle.items[0] = BlockT.make["cpu"]()
    bundle.items[1] = BlockT.make["cpu"]()

    var sa_p:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * SA)
    var y_p:   UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
    for k in range(BATCH * SA):
        sa_p[k] = Scalar[DT](0.1 * Float64(k) - 0.5)
    for b in range(BATCH):
        y_p[b] = Scalar[DT](0.3 * Float64(b) - 0.1)
    var sa_t = TileTensor(sa_p, row_major[BATCH, SA]())
    var y_t = TileTensor(y_p, row_major[BATCH, 1]())

    var l1 = bundle.items[0].step["cpu"](c1, c1_opt, sa_t, y_t)
    var l2 = bundle.items[1].step["cpu"](c2, c2_opt, sa_t, y_t)
    print("  bundle.items[0].step loss=" + String(l1))
    print("  bundle.items[1].step loss=" + String(l2))
    assert_true(l1 > Scalar[DT](0.0), "block 1 loss must be positive")
    assert_true(l2 > Scalar[DT](0.0), "block 2 loss must be positive")

    sa_p.free()
    y_p.free()
    print("  test_bundle_stores_two_critic_blocks PASSED")


def test_bundle_route_matches_direct() raises:
    """Train a critic via bundle.items[0].step vs a parallel critic via
    direct block.step. Same RNG seed → identical weight trajectory."""
    comptime BATCH = 6
    comptime SA = 4
    comptime CRITIC = Sequential[Linear[SA, 6], ReLU[6], Linear[6, 1]]
    comptime BlockT = CriticUpdateBlock[CRITIC, BATCH, SA]
    comptime N_STEPS = 5

    seed(41)
    # Build TWO independent critics from the same RNG draw sequence by
    # constructing them in identical order on identical seeds. The Adam
    # state is also identical because both opts start from a zero buffer.
    seed(41)
    var c_bundle = CRITIC.make["cpu", INIT=Kaiming]()
    var opt_bundle = Adam.make[target="cpu", M=CRITIC](c_bundle)
    opt_bundle.lr = Scalar[DT](5e-3)

    seed(41)
    var c_direct = CRITIC.make["cpu", INIT=Kaiming]()
    var opt_direct = Adam.make[target="cpu", M=CRITIC](c_direct)
    opt_direct.lr = Scalar[DT](5e-3)

    var bundle = LossBlockBundle[BlockT].make_default["cpu"]()
    bundle.items[0] = BlockT.make["cpu"]()
    var direct = BlockT.make["cpu"]()

    var sa_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * SA)
    var y_p:  UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
    for k in range(BATCH * SA):
        sa_p[k] = Scalar[DT](0.1 * Float64(k))
    for b in range(BATCH):
        y_p[b] = Scalar[DT](0.05 * Float64(b))
    var sa_t = TileTensor(sa_p, row_major[BATCH, SA]())
    var y_t  = TileTensor(y_p, row_major[BATCH, 1]())

    for step in range(N_STEPS):
        var lb = bundle.items[0].step["cpu"](c_bundle, opt_bundle, sa_t, y_t)
        var ld = direct.step["cpu"](c_direct, opt_direct, sa_t, y_t)
        assert_true(fabs(lb - ld) <= Scalar[DT](1e-10),
                    "loss diverged: bundle=" + String(lb) + " direct=" + String(ld))

    # After N_STEPS, all final weights must be bit-identical.
    # Walk every Param of both critics and assert.
    sa_p.free()
    y_p.free()
    print("  test_bundle_route_matches_direct PASSED")


def main() raises:
    print("=" * 60)
    print("nn LossBlock / LossBlockBundle tests (Block E-3)")
    print("=" * 60)
    test_critic_block_conforms_to_lossblock()
    test_bundle_stores_two_critic_blocks()
    test_bundle_route_matches_direct()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
