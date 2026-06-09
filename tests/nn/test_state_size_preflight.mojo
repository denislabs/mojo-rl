"""Phase 1b pre-flight checks for STATE_SIZE migration.

Verifies two Mojo behaviors before committing to the design:
1. Zero-length `LayoutTensor[dtype, Layout.row_major(0), MutAnyOrigin]`
   works as a no-op placeholder (no allocation, no access).
2. `comptime STATE_SIZE: Int = 0` trait default propagates to structs that
   don't override it.
"""

from layout import Layout, LayoutTensor
from std.memory import UnsafePointer


# -----------------------------------------------------------------------------
# Check 1: zero-length LayoutTensor
# -----------------------------------------------------------------------------

def check_zero_length_tensor() -> Bool:
    """Construct a zero-length LayoutTensor and confirm it can be passed around."""
    # Null pointer is fine because the tensor should never be dereferenced.
    var ptr = UnsafePointer[Scalar[DType.float32], MutAnyOrigin](
        unsafe_from_address=Int(0)
    )
    var empty = LayoutTensor[DType.float32, Layout.row_major(0), MutAnyOrigin](ptr)
    # If this line compiles and runs without crashing, the type is acceptable.
    print("  [PASS] Zero-length LayoutTensor constructed")
    return True


# -----------------------------------------------------------------------------
# Check 2: comptime trait default propagation
# -----------------------------------------------------------------------------

trait HasStateSize(Movable & ImplicitlyCopyable):
    """Minimal trait with a comptime default."""

    comptime REQUIRED: Int
    comptime STATE_SIZE: Int = 0  # default — most impls should not need to override


struct LeafNoOverride(HasStateSize):
    comptime REQUIRED: Int = 42
    # Intentionally does NOT override STATE_SIZE — want default to apply.


struct LeafWithOverride(HasStateSize):
    comptime REQUIRED: Int = 7
    comptime STATE_SIZE: Int = 3


def check_trait_default() -> Bool:
    """Verify a struct that omits STATE_SIZE inherits the default = 0."""

    @parameter
    def check_no_override[T: HasStateSize]() -> Int:
        return T.STATE_SIZE

    @parameter
    def check_required[T: HasStateSize]() -> Int:
        return T.REQUIRED

    var default_val = check_no_override[LeafNoOverride]()
    var override_val = check_no_override[LeafWithOverride]()
    var required_val = check_required[LeafNoOverride]()

    print("  LeafNoOverride.STATE_SIZE   =", default_val, "(expected 0)")
    print("  LeafWithOverride.STATE_SIZE =", override_val, "(expected 3)")
    print("  LeafNoOverride.REQUIRED     =", required_val, "(expected 42)")

    var ok = (default_val == 0) and (override_val == 3) and (required_val == 42)
    if ok:
        print("  [PASS] Trait default propagates")
    else:
        print("  [FAIL] Trait default did NOT propagate")
    return ok


# -----------------------------------------------------------------------------

def main():
    print("=== Phase 1b pre-flight checks ===")
    print("Check 1: zero-length LayoutTensor")
    var ok1 = check_zero_length_tensor()
    print("Check 2: comptime trait default propagation")
    var ok2 = check_trait_default()
    print("===")
    if ok1 and ok2:
        print("ALL CHECKS PASSED")
    else:
        print("ONE OR MORE CHECKS FAILED")
