//! Compiled SQL `LIKE` patterns.
//!
//! Most real-world patterns are of the form `'abc%'`, `'%abc'` or `'%abc%'`.
//! We recognize those shapes once at plan time and use `starts_with` /
//! `ends_with` / substring search instead of a general matcher.

use std::fmt;

/// A LIKE pattern, specialized by shape.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LikePattern {
    /// no wildcards: plain equality
    Exact(Vec<u8>),
    /// `abc%`
    Prefix(Vec<u8>),
    /// `%abc`
    Suffix(Vec<u8>),
    /// `%abc%`
    Contains(Vec<u8>),
    /// anything else (`_`, several `%` groups ...), matched by backtracking
    General(Vec<u8>),
}

impl LikePattern {
    /// Compile a SQL LIKE pattern (no ESCAPE clause support).
    pub fn compile(pattern: &str) -> LikePattern {
        let p = pattern.as_bytes();
        let has_underscore = p.contains(&b'_');
        let inner = |s: &[u8]| !s.contains(&b'%');
        if !has_underscore {
            let n = p.len();
            if !p.contains(&b'%') {
                return LikePattern::Exact(p.to_vec());
            }
            if n >= 2 && p[0] == b'%' && p[n - 1] == b'%' && inner(&p[1..n - 1]) {
                return LikePattern::Contains(p[1..n - 1].to_vec());
            }
            if p[n - 1] == b'%' && inner(&p[..n - 1]) {
                return LikePattern::Prefix(p[..n - 1].to_vec());
            }
            if p[0] == b'%' && inner(&p[1..]) {
                return LikePattern::Suffix(p[1..].to_vec());
            }
        }
        LikePattern::General(p.to_vec())
    }

    /// Does `s` match?
    #[inline]
    pub fn matches(&self, s: &[u8]) -> bool {
        match self {
            LikePattern::Exact(p) => s == p.as_slice(),
            LikePattern::Prefix(p) => s.starts_with(p),
            LikePattern::Suffix(p) => s.ends_with(p),
            LikePattern::Contains(p) => contains(s, p),
            LikePattern::General(p) => like_general(s, p),
        }
    }
}

/// Substring search. A simple first-byte scan followed by a slice compare is
/// fast for the short needles typical in SQL predicates.
pub fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    if needle.is_empty() {
        return true;
    }
    if needle.len() > haystack.len() {
        return false;
    }
    let first = needle[0];
    let last_start = haystack.len() - needle.len();
    let mut i = 0;
    while i <= last_start {
        match haystack[i..=last_start].iter().position(|&b| b == first) {
            None => return false,
            Some(off) => {
                let start = i + off;
                if &haystack[start..start + needle.len()] == needle {
                    return true;
                }
                i = start + 1;
            }
        }
    }
    false
}

/// General LIKE with `%` (any run) and `_` (one byte), using the classic
/// two-pointer algorithm with a single backtrack point: O(n*m) worst case,
/// linear for typical patterns.
fn like_general(s: &[u8], p: &[u8]) -> bool {
    let (mut si, mut pi) = (0usize, 0usize);
    let mut star: Option<usize> = None;
    let mut star_s = 0usize;
    while si < s.len() {
        if pi < p.len() && (p[pi] == b'_' || p[pi] == s[si]) {
            si += 1;
            pi += 1;
        } else if pi < p.len() && p[pi] == b'%' {
            star = Some(pi);
            star_s = si;
            pi += 1;
        } else if let Some(sp) = star {
            pi = sp + 1;
            star_s += 1;
            si = star_s;
        } else {
            return false;
        }
    }
    while pi < p.len() && p[pi] == b'%' {
        pi += 1;
    }
    pi == p.len()
}

impl fmt::Display for LikePattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = |b: &[u8]| String::from_utf8_lossy(b).into_owned();
        match self {
            LikePattern::Exact(p) => write!(f, "'{}'", s(p)),
            LikePattern::Prefix(p) => write!(f, "'{}%'", s(p)),
            LikePattern::Suffix(p) => write!(f, "'%{}'", s(p)),
            LikePattern::Contains(p) => write!(f, "'%{}%'", s(p)),
            LikePattern::General(p) => write!(f, "'{}'", s(p)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shapes() {
        assert_eq!(
            LikePattern::compile("PROMO%"),
            LikePattern::Prefix(b"PROMO".to_vec())
        );
        assert_eq!(
            LikePattern::compile("%BRASS"),
            LikePattern::Suffix(b"BRASS".to_vec())
        );
        assert_eq!(
            LikePattern::compile("%green%"),
            LikePattern::Contains(b"green".to_vec())
        );
        assert!(matches!(
            LikePattern::compile("%a%b%"),
            LikePattern::General(_)
        ));
    }

    #[test]
    fn matching() {
        let m = |p: &str, s: &str| LikePattern::compile(p).matches(s.as_bytes());
        assert!(m("PROMO%", "PROMO BRUSHED"));
        assert!(!m("PROMO%", "STANDARD"));
        assert!(m("%BRASS", "LARGE BRASS"));
        assert!(m("%green%", "forest green metallic"));
        assert!(m("%special%requests%", "a special big requests x"));
        assert!(!m("%special%requests%", "requests special"));
        assert!(m("a_c", "abc"));
        assert!(!m("a_c", "abbc"));
        assert!(m("%", ""));
    }
}
