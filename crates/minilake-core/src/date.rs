//! Calendar arithmetic for `DATE` values stored as days since 1970-01-01.
//!
//! Uses Howard Hinnant's `days_from_civil` / `civil_from_days` algorithms,
//! which are branch-light and exact over the proleptic Gregorian calendar.

/// Days since the epoch for a (year, month 1-12, day 1-31) triple.
pub fn days_from_civil(y: i32, m: u32, d: u32) -> i32 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = (y - era * 400) as u32; // [0, 399]
    let mp = (m + 9) % 12; // March = 0
    let doy = (153 * mp + 2) / 5 + d - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    era * 146_097 + doe as i32 - 719_468
}

/// Inverse of [`days_from_civil`].
pub fn civil_from_days(z: i32) -> (i32, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i32 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// Parse `YYYY-MM-DD`.
pub fn parse_date(s: &str) -> Option<i32> {
    let mut parts = s.trim().split('-');
    let y: i32 = parts.next()?.parse().ok()?;
    let m: u32 = parts.next()?.parse().ok()?;
    let d: u32 = parts.next()?.parse().ok()?;
    if parts.next().is_some() || !(1..=12).contains(&m) || d == 0 || d > days_in_month(y, m) {
        return None;
    }
    Some(days_from_civil(y, m, d))
}

/// Format as `YYYY-MM-DD`.
pub fn format_date(days: i32) -> String {
    let (y, m, d) = civil_from_days(days);
    format!("{y:04}-{m:02}-{d:02}")
}

fn is_leap(y: i32) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

/// Number of days in the given month.
pub fn days_in_month(y: i32, m: u32) -> u32 {
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap(y) => 29,
        _ => 28,
    }
}

/// Add a number of calendar months, clamping the day (Jan 31 + 1 month = Feb 28/29),
/// which matches PostgreSQL and DuckDB semantics.
pub fn add_months(days: i32, months: i32) -> i32 {
    let (y, m, d) = civil_from_days(days);
    let total = y * 12 + (m as i32 - 1) + months;
    let ny = total.div_euclid(12);
    let nm = total.rem_euclid(12) as u32 + 1;
    let nd = d.min(days_in_month(ny, nm));
    days_from_civil(ny, nm, nd)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        for z in -800_000..800_000 {
            let (y, m, d) = civil_from_days(z);
            assert_eq!(days_from_civil(y, m, d), z);
        }
    }

    #[test]
    fn known_dates() {
        assert_eq!(parse_date("1970-01-01"), Some(0));
        assert_eq!(format_date(parse_date("1998-12-01").unwrap()), "1998-12-01");
        assert_eq!(
            add_months(parse_date("1994-01-31").unwrap(), 1),
            parse_date("1994-02-28").unwrap()
        );
        assert_eq!(parse_date("1994-02-30"), None);
    }
}
