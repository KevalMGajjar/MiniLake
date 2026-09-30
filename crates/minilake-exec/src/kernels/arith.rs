//! Arithmetic kernels.
//!
//! Integer kernels use wrapping arithmetic: a checked add would introduce a
//! branch per element and block vectorization. TPC-H values are far from the
//! i64 limits; overflow semantics are documented as a known difference from
//! DuckDB (which raises an error).

use minilake_core::{MiniLakeError, Result};

use super::{map2, Operand};
use crate::expr::BinaryOp;

/// Numeric element types the arithmetic kernels support.
pub trait Numeric: Copy + PartialEq + Default + Send + Sync + 'static {
    /// a + b
    fn add(self, o: Self) -> Self;
    /// a - b
    fn sub(self, o: Self) -> Self;
    /// a * b
    fn mul(self, o: Self) -> Self;
    /// a % b (0 when b == 0; the caller masks those rows as NULL)
    fn rem(self, o: Self) -> Self;
    /// Is this the zero value?
    fn is_zero(self) -> bool {
        self == Self::default()
    }
}

impl Numeric for i32 {
    #[inline]
    fn add(self, o: Self) -> Self {
        self.wrapping_add(o)
    }
    #[inline]
    fn sub(self, o: Self) -> Self {
        self.wrapping_sub(o)
    }
    #[inline]
    fn mul(self, o: Self) -> Self {
        self.wrapping_mul(o)
    }
    #[inline]
    fn rem(self, o: Self) -> Self {
        if o == 0 {
            0
        } else {
            self.wrapping_rem(o)
        }
    }
}

impl Numeric for i64 {
    #[inline]
    fn add(self, o: Self) -> Self {
        self.wrapping_add(o)
    }
    #[inline]
    fn sub(self, o: Self) -> Self {
        self.wrapping_sub(o)
    }
    #[inline]
    fn mul(self, o: Self) -> Self {
        self.wrapping_mul(o)
    }
    #[inline]
    fn rem(self, o: Self) -> Self {
        if o == 0 {
            0
        } else {
            self.wrapping_rem(o)
        }
    }
}

impl Numeric for f64 {
    #[inline]
    fn add(self, o: Self) -> Self {
        self + o
    }
    #[inline]
    fn sub(self, o: Self) -> Self {
        self - o
    }
    #[inline]
    fn mul(self, o: Self) -> Self {
        self * o
    }
    #[inline]
    fn rem(self, o: Self) -> Self {
        self % o
    }
}

/// `out[i] = l[i] op r[i]` for `+ - * %`.
pub fn arith<T: Numeric>(
    op: BinaryOp,
    l: Operand<'_, T>,
    r: Operand<'_, T>,
    len: usize,
) -> Result<Vec<T>> {
    Ok(match op {
        BinaryOp::Add => map2(l, r, len, T::add),
        BinaryOp::Sub => map2(l, r, len, T::sub),
        BinaryOp::Mul => map2(l, r, len, T::mul),
        BinaryOp::Mod => map2(l, r, len, T::rem),
        other => {
            return Err(MiniLakeError::Internal(format!(
                "{other:?} is not an integer/float arithmetic op"
            )))
        }
    })
}

/// Floating-point division `out[i] = l[i] / r[i]`.
pub fn div_f64(l: Operand<'_, f64>, r: Operand<'_, f64>, len: usize) -> Vec<f64> {
    map2(l, r, len, |a, b| a / b)
}

/// Plain `f64` kernels exposed for benchmarks and assembly inspection
/// (`cargo asm -p minilake-exec minilake_exec::kernels::arith::mul_f64`).
#[inline(never)]
pub fn mul_f64(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x * y).collect()
}

/// `a[i] * (1 - b[i])`, the TPC-H "discounted price" expression, fused.
#[inline(never)]
pub fn mul_one_minus_f64(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x * (1.0 - y)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arith_ops() {
        let a = [1i64, 2, 3];
        let b = [10i64, 20, 30];
        assert_eq!(
            arith(BinaryOp::Add, Operand::Slice(&a), Operand::Slice(&b), 3).unwrap(),
            vec![11, 22, 33]
        );
        assert_eq!(
            arith(BinaryOp::Sub, Operand::Scalar(100), Operand::Slice(&b), 3).unwrap(),
            vec![90, 80, 70]
        );
        assert_eq!(
            arith(BinaryOp::Mod, Operand::Slice(&b), Operand::Scalar(0), 3).unwrap(),
            vec![0, 0, 0]
        );
        assert_eq!(mul_one_minus_f64(&[10.0], &[0.1]), vec![9.0]);
    }
}
