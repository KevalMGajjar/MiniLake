//! Validity bitmap: bit `i` is 1 when row `i` is NOT null.
//!
//! Bits are packed into `u64` words (LSB first), the same convention Arrow and
//! DuckDB use. A column without a bitmap has no nulls at all, which lets the
//! hot kernels skip null handling entirely.

/// A packed bitmap of `len` bits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Bitmap {
    words: Vec<u64>,
    len: usize,
}

impl Bitmap {
    /// All bits set (all rows valid).
    pub fn new_set(len: usize) -> Self {
        let mut b = Bitmap {
            words: vec![u64::MAX; len.div_ceil(64)],
            len,
        };
        b.clear_tail();
        b
    }

    /// All bits unset (all rows null).
    pub fn new_unset(len: usize) -> Self {
        Bitmap {
            words: vec![0; len.div_ceil(64)],
            len,
        }
    }

    /// Build from a slice of bools.
    pub fn from_bools(bits: &[bool]) -> Self {
        let mut b = Bitmap::new_unset(bits.len());
        for (i, &v) in bits.iter().enumerate() {
            if v {
                b.words[i / 64] |= 1 << (i % 64);
            }
        }
        b
    }

    fn clear_tail(&mut self) {
        let rem = self.len % 64;
        if rem != 0 {
            if let Some(last) = self.words.last_mut() {
                *last &= (1u64 << rem) - 1;
            }
        }
    }

    /// Number of bits.
    pub fn len(&self) -> usize {
        self.len
    }

    /// True when the bitmap has zero bits.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Read bit `i`.
    #[inline]
    pub fn get(&self, i: usize) -> bool {
        debug_assert!(i < self.len);
        (self.words[i / 64] >> (i % 64)) & 1 == 1
    }

    /// Write bit `i`.
    #[inline]
    pub fn set(&mut self, i: usize, v: bool) {
        let w = &mut self.words[i / 64];
        let mask = 1u64 << (i % 64);
        if v {
            *w |= mask;
        } else {
            *w &= !mask;
        }
    }

    /// Append one bit.
    pub fn push(&mut self, v: bool) {
        if self.len % 64 == 0 {
            self.words.push(0);
        }
        self.len += 1;
        let i = self.len - 1;
        self.set(i, v);
    }

    /// Number of set bits (valid rows).
    pub fn count_set(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Bitwise AND; both bitmaps must have the same length.
    pub fn and(&self, other: &Bitmap) -> Bitmap {
        debug_assert_eq!(self.len, other.len);
        Bitmap {
            words: self
                .words
                .iter()
                .zip(&other.words)
                .map(|(a, b)| a & b)
                .collect(),
            len: self.len,
        }
    }

    /// Keep only the bits at `indices`, in order.
    pub fn gather(&self, indices: &[u32]) -> Bitmap {
        let mut out = Bitmap::new_unset(indices.len());
        for (o, &i) in indices.iter().enumerate() {
            if self.get(i as usize) {
                out.words[o / 64] |= 1 << (o % 64);
            }
        }
        out
    }

    /// Bits `[offset, offset + len)`.
    pub fn slice(&self, offset: usize, len: usize) -> Bitmap {
        let mut out = Bitmap::new_unset(len);
        for o in 0..len {
            if self.get(offset + o) {
                out.words[o / 64] |= 1 << (o % 64);
            }
        }
        out
    }

    /// The raw words (for serialization).
    pub fn words(&self) -> &[u64] {
        &self.words
    }

    /// Rebuild from raw words.
    pub fn from_words(words: Vec<u64>, len: usize) -> Self {
        let mut b = Bitmap { words, len };
        b.words.resize(len.div_ceil(64), 0);
        b.clear_tail();
        b
    }

    /// Heap bytes used.
    pub fn memory_size(&self) -> usize {
        self.words.len() * 8
    }
}

/// AND two optional validity bitmaps (`None` means "all valid").
pub fn combine_validity(a: Option<&Bitmap>, b: Option<&Bitmap>) -> Option<Bitmap> {
    match (a, b) {
        (None, None) => None,
        (Some(x), None) | (None, Some(x)) => Some(x.clone()),
        (Some(x), Some(y)) => Some(x.and(y)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_ops() {
        let mut b = Bitmap::new_set(70);
        assert_eq!(b.count_set(), 70);
        b.set(3, false);
        b.set(69, false);
        assert!(!b.get(3) && !b.get(69) && b.get(68));
        assert_eq!(b.count_set(), 68);
        let g = b.gather(&[3, 4, 69]);
        assert_eq!((g.get(0), g.get(1), g.get(2)), (false, true, false));
        b.push(true);
        assert_eq!(b.len(), 71);
        assert!(b.get(70));
    }
}
