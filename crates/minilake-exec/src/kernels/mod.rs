//! Vectorized kernels: tight, type-specialized loops over whole columns.
//!
//! Design rules followed by every kernel in this module:
//!
//! 1. **One loop per (type, operator) pair.** Kernels are generic over the
//!    element type and take the operator as a closure; each call site is
//!    monomorphized, so LLVM sees e.g. `a[i] < s` over `&[f64]` with no
//!    dynamic dispatch inside the loop and can auto-vectorize it (NEON on
//!    aarch64, SSE/AVX on x86-64).
//! 2. **No branches on NULL inside the loop.** Values at null positions are
//!    computed like any other value and the validity bitmap is combined
//!    separately.
//! 3. **Iterator `zip`/`map`/`collect` over slices.** This gives LLVM exact
//!    trip counts (no bounds checks) and pre-sized output vectors.
//!
//! See `docs/PERF_LOG.md` for how the generated assembly was inspected.

pub mod arith;
pub mod boolean;
pub mod cast;
pub mod cmp;
pub mod hash;
pub mod scalar_ops;
pub mod string;

/// One side of a binary kernel: a column slice or a broadcast constant.
///
/// Keeping `Scalar` separate avoids materializing a full column for a literal
/// (as in `l_quantity < 24`) and lets the kernel keep the constant in a
/// register.
#[derive(Clone, Copy, Debug)]
pub enum Operand<'a, T> {
    /// one value per row
    Slice(&'a [T]),
    /// the same value for every row
    Scalar(T),
}

/// Apply `f` element-wise to two operands of the same length `len`.
#[inline]
pub fn map2<T: Copy, U: Copy, F: Fn(T, T) -> U>(
    l: Operand<'_, T>,
    r: Operand<'_, T>,
    len: usize,
    f: F,
) -> Vec<U> {
    match (l, r) {
        (Operand::Slice(a), Operand::Slice(b)) => {
            a.iter().zip(b.iter()).map(|(&x, &y)| f(x, y)).collect()
        }
        (Operand::Slice(a), Operand::Scalar(s)) => a.iter().map(|&x| f(x, s)).collect(),
        (Operand::Scalar(s), Operand::Slice(b)) => b.iter().map(|&y| f(s, y)).collect(),
        (Operand::Scalar(x), Operand::Scalar(y)) => vec![f(x, y); len],
    }
}
