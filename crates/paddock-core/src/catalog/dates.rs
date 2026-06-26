//! Minimal civil-calendar date helpers (UTC, day precision). Hand-rolled to
//! avoid a chrono dependency: the catalog only needs "parse a date string to
//! epoch seconds" and "subtract relative ages".

/// Days from civil date to 1970-01-01 (Howard Hinnant's algorithm), then to
/// epoch seconds. Returns None for out-of-range month/day. Years are bounded
/// to a sane catalog range: years outside 1970..=9999 → None (also keeps the
/// arithmetic far from i64 overflow on hostile input).
pub fn ymd_to_epoch(y: i64, m: u32, d: u32) -> Option<i64> {
    if !(1970..=9999).contains(&y) {
        return None;
    }
    if !(1..=12).contains(&m) {
        return None;
    }
    let leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
    let dim = [
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    if d < 1 || d > dim[(m - 1) as usize] {
        return None;
    }
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m as i64 + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    Some((era * 146_097 + doe - 719_468) * 86_400)
}

/// `"YYYY-MM"` → epoch of the first of that month (curated JSON format).
pub fn parse_year_month(s: &str) -> Option<i64> {
    let (y, m) = s.split_once('-')?;
    if y.len() != 4 || m.len() != 2 {
        return None;
    }
    ymd_to_epoch(y.parse().ok()?, m.parse().ok()?, 1)
}

/// ISO-8601 day prefix (`"2024-03-07T…"` or `"2024-03-07"`) → epoch.
/// Strict `YYYY-MM-DD` shape (HF always zero-pads).
pub fn parse_iso_date_prefix(s: &str) -> Option<i64> {
    let s = s.get(..10)?;
    let b = s.as_bytes();
    if b[4] != b'-' || b[7] != b'-' {
        return None;
    }
    ymd_to_epoch(
        s[..4].parse().ok()?,
        s[5..7].parse().ok()?,
        s[8..10].parse().ok()?,
    )
}

/// `"N years/months/weeks/days ago"` (and `"yesterday"`, sub-day units) →
/// epoch relative to `now`. Month = 30 d, year = 365 d - display precision.
pub fn parse_relative_ago(s: &str, now: i64) -> Option<i64> {
    const DAY: i64 = 86_400;
    let s = s.trim().to_lowercase();
    if s == "yesterday" {
        return Some(now - DAY);
    }
    let mut it = s.split_whitespace();
    let n: i64 = it.next()?.parse().ok()?;
    if n < 0 {
        return None; // "-3 days ago" would be a future date
    }
    let unit = it.next()?;
    if it.next() != Some("ago") {
        return None;
    }
    let unit = unit.strip_suffix('s').unwrap_or(unit);
    let days = match unit {
        "day" => n,
        "week" => n.checked_mul(7)?,
        "month" => n.checked_mul(30)?,
        "year" => n.checked_mul(365)?,
        "hour" | "minute" | "second" => 0,
        _ => return None,
    };
    now.checked_sub(days.checked_mul(DAY)?)
}

/// Scan a page for every `"N unit ago"` mention and return the OLDEST as an
/// epoch. Used on Ollama tags pages where the oldest tag date is the closest
/// available proxy for the release date (a full re-push refreshes all of
/// them - that failure mode is accepted; see the design spec).
pub fn oldest_relative_date(html: &str, now: i64) -> Option<i64> {
    let mut oldest: Option<i64> = None;
    for (idx, _) in html.match_indices(" ago") {
        // Require a word boundary after "ago" so "3 days agonizing" doesn't
        // count: the next byte (if any) must not be alphanumeric.
        if let Some(&b) = html.as_bytes().get(idx + 4)
            && b.is_ascii_alphanumeric()
        {
            continue;
        }
        // Walk back over "N unit " (max ~20 chars: "59 minutes").
        let start = html[..idx]
            .char_indices()
            .rev()
            .take(20)
            .map(|(i, _)| i)
            .last()
            .unwrap_or(0);
        let window = &html[start..idx + 4];
        // Try every suffix of the window that starts at a digit.
        for (i, c) in window.char_indices() {
            if c.is_ascii_digit()
                && let Some(e) = parse_relative_ago(&window[i..], now)
            {
                oldest = Some(oldest.map_or(e, |o| o.min(e)));
                break;
            }
        }
    }
    oldest
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ymd_to_epoch_known_values() {
        assert_eq!(ymd_to_epoch(1970, 1, 1), Some(0));
        assert_eq!(ymd_to_epoch(2024, 7, 23), Some(1_721_692_800)); // llama3.1 day
        assert_eq!(ymd_to_epoch(2025, 4, 1), Some(1_743_465_600));
        assert_eq!(ymd_to_epoch(2024, 2, 29), Some(1_709_164_800)); // leap day
        assert_eq!(ymd_to_epoch(2024, 13, 1), None);
        assert_eq!(ymd_to_epoch(2024, 0, 1), None);
        assert_eq!(ymd_to_epoch(2024, 2, 30), None);
    }

    #[test]
    fn ymd_to_epoch_rejects_out_of_range_years() {
        assert_eq!(ymd_to_epoch(i64::MAX, 1, 1), None);
        assert_eq!(ymd_to_epoch(i64::MIN, 1, 1), None);
        assert_eq!(ymd_to_epoch(1969, 12, 31), None);
        assert_eq!(ymd_to_epoch(10_000, 1, 1), None);
        assert_eq!(ymd_to_epoch(9999, 12, 31), Some(253_402_214_400));
    }

    #[test]
    fn parse_year_month_cases() {
        assert_eq!(parse_year_month("2025-04"), ymd_to_epoch(2025, 4, 1));
        assert_eq!(parse_year_month("2024-12"), ymd_to_epoch(2024, 12, 1));
        assert_eq!(parse_year_month("2025"), None);
        assert_eq!(parse_year_month("2025-13"), None);
        assert_eq!(parse_year_month("avril 2025"), None);
    }

    #[test]
    fn parse_iso_date_prefix_cases() {
        // HF createdAt: "2024-03-07T15:45:34.000Z" - day prefix is enough.
        assert_eq!(
            parse_iso_date_prefix("2024-03-07T15:45:34.000Z"),
            ymd_to_epoch(2024, 3, 7)
        );
        assert_eq!(
            parse_iso_date_prefix("2024-03-07"),
            ymd_to_epoch(2024, 3, 7)
        );
        assert_eq!(parse_iso_date_prefix("garbage"), None);
        assert_eq!(parse_iso_date_prefix("2024-3-7"), None);
    }

    #[test]
    fn parse_relative_ago_cases() {
        const NOW: i64 = 1_780_000_000;
        const DAY: i64 = 86_400;
        assert_eq!(parse_relative_ago("3 days ago", NOW), Some(NOW - 3 * DAY));
        assert_eq!(parse_relative_ago("1 day ago", NOW), Some(NOW - DAY));
        assert_eq!(parse_relative_ago("2 weeks ago", NOW), Some(NOW - 14 * DAY));
        assert_eq!(
            parse_relative_ago("6 months ago", NOW),
            Some(NOW - 6 * 30 * DAY)
        );
        assert_eq!(parse_relative_ago("1 year ago", NOW), Some(NOW - 365 * DAY));
        assert_eq!(parse_relative_ago("yesterday", NOW), Some(NOW - DAY));
        assert_eq!(parse_relative_ago("5 hours ago", NOW), Some(NOW)); // < 1 day → now
        assert_eq!(parse_relative_ago("soon", NOW), None);
    }

    #[test]
    fn parse_relative_ago_rejects_hostile_input() {
        const NOW: i64 = 1_780_000_000;
        // Huge n must not overflow (panic in debug, wrap in release).
        assert_eq!(parse_relative_ago("99999999999999999 years ago", NOW), None);
        assert_eq!(
            parse_relative_ago("9223372036854775807 days ago", NOW),
            None
        );
        // Negative n would produce a future date.
        assert_eq!(parse_relative_ago("-3 days ago", NOW), None);
    }

    #[test]
    fn oldest_relative_date_scans_whole_page() {
        const NOW: i64 = 1_780_000_000;
        let html = r#"
            <span x-test-updated>2 months ago</span>
            <div>… 46e0c10c039e&nbsp;·&nbsp;1 year ago</div>
            <span>3 weeks ago</span>
        "#;
        // Oldest mention wins: 1 year ago.
        assert_eq!(oldest_relative_date(html, NOW), Some(NOW - 365 * 86_400));
        assert_eq!(oldest_relative_date("<html>no dates</html>", NOW), None);
    }

    #[test]
    fn oldest_relative_date_requires_word_boundary_after_ago() {
        const NOW: i64 = 1_780_000_000;
        const DAY: i64 = 86_400;
        // " ago" inside a longer word is not a date.
        assert_eq!(
            oldest_relative_date("we waited 3 days agonizing over it", NOW),
            None
        );
        // Punctuation or markup right after "ago" is still a date.
        assert_eq!(
            oldest_relative_date("pushed 3 days ago.", NOW),
            Some(NOW - 3 * DAY)
        );
        assert_eq!(
            oldest_relative_date("<span>3 days ago</span>", NOW),
            Some(NOW - 3 * DAY)
        );
        // End of string right after "ago" is fine too.
        assert_eq!(oldest_relative_date("3 days ago", NOW), Some(NOW - 3 * DAY));
    }

    #[test]
    fn oldest_relative_date_survives_hostile_numbers() {
        const NOW: i64 = 1_780_000_000;
        // Must not panic (debug overflow) on absurd counts. The 20-char window
        // may salvage a smaller "N years ago" digit suffix, but the result must
        // never be a wrapped future date.
        let html = "<span>99999999999999999 years ago</span>";
        let got = oldest_relative_date(html, NOW);
        assert!(got.is_none_or(|e| e <= NOW));
    }
}
