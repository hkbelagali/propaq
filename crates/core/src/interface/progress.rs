//!
//! Progress bar for a propagation run.
//!

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Milliseconds between draws, whatever the gate rate.
const MIN_DRAW_INTERVAL_MS: u64 = 100;

/// A tqdm bar, driven from inside the detached gate loop.
pub struct Progress {
    bar: Py<PyAny>,
    postfix: Py<PyDict>,
    /// Gates between ticks.
    every: usize,
    start: Instant,
    /// Milliseconds since `start` at the last draw.
    last_draw_ms: AtomicU64,
    /// Gates advanced but not yet shown.
    pending: AtomicUsize,
    /// Whether the bar has yet to draw anything.
    first: AtomicBool,
}

impl Progress {
    /// Builds a bar over `total` gates, or `None` when progress is off.
    pub fn new(
        py: Python<'_>,
        enabled: bool,
        total: usize,
        every: usize,
    ) -> PyResult<Option<Self>> {
        if !enabled {
            return Ok(None);
        }
        let kwargs = PyDict::new(py);
        kwargs.set_item("total", total)?;
        kwargs.set_item("desc", "Propagating")?;
        kwargs.set_item("unit", "gate")?;
        let bar = py
            .import("tqdm.auto")?
            .call_method("tqdm", (), Some(&kwargs))?;
        Ok(Some(Progress {
            bar: bar.into(),
            postfix: PyDict::new(py).into(),
            every: every.max(1),
            start: Instant::now(),
            last_draw_ms: AtomicU64::new(0),
            pending: AtomicUsize::new(0),
            first: AtomicBool::new(true),
        }))
    }

    /// Gates between ticks, never zero.
    pub fn every(&self) -> usize {
        self.every
    }

    /// Advances by `advance` gates, showing the live term count.
    pub fn tick(&self, advance: usize, n_terms: usize) {
        let Some(advance) = self.bank(advance) else {
            return;
        };
        Python::attach(|py| {
            let _ = self.draw(py, advance, |postfix| {
                postfix.set_item("terms", n_terms)?;
                Ok(())
            });
        });
    }

    /// Advances by `advance` gates, showing terms and the surrogate's monomial
    /// count.
    pub fn tick_surrogate(&self, advance: usize, n_terms: usize, monomials: u128) {
        let Some(advance) = self.bank(advance) else {
            return;
        };
        Python::attach(|py| {
            let _ = self.draw(py, advance, |postfix| {
                postfix.set_item("terms", n_terms)?;
                postfix.set_item("mono~", compact(monomials))?;
                Ok(())
            });
        });
    }

    /// Flushes the banked gates and closes the bar, releasing its terminal line.
    pub fn close(&self) {
        Python::attach(|py| {
            let pending = self.pending.swap(0, Ordering::Relaxed);
            if pending > 0 {
                let _ = self.bar.bind(py).call_method1("update", (pending,));
            }
            let _ = self.bar.bind(py).call_method0("close");
        });
    }

    /// Banks `advance` gates, returning the total to draw when one is due.
    fn bank(&self, advance: usize) -> Option<usize> {
        self.pending.fetch_add(advance, Ordering::Relaxed);
        let now_ms = self.start.elapsed().as_millis() as u64;
        // The first tick always draws, so a bar shows real content straight away
        // instead of sitting blank for the first tenth of a second.
        let due = self.first.swap(false, Ordering::Relaxed)
            || now_ms.saturating_sub(self.last_draw_ms.load(Ordering::Relaxed))
                >= MIN_DRAW_INTERVAL_MS;
        if !due {
            return None;
        }
        self.last_draw_ms.store(now_ms, Ordering::Relaxed);
        let drawn = self.pending.swap(0, Ordering::Relaxed);
        (drawn > 0).then_some(drawn)
    }

    fn draw<F>(&self, py: Python<'_>, advance: usize, fill: F) -> PyResult<()>
    where
        F: FnOnce(&Bound<'_, PyDict>) -> PyResult<()>,
    {
        let postfix = self.postfix.bind(py);
        fill(postfix)?;
        let bar = self.bar.bind(py);

        let kwargs = PyDict::new(py);
        kwargs.set_item("refresh", false)?;
        bar.call_method("set_postfix", (postfix,), Some(&kwargs))?;
        bar.call_method1("update", (advance,))?;
        Ok(())
    }
}

/// Renders a count in SI-ish short form.
fn compact(n: u128) -> String {
    const UNITS: [&str; 12] = ["", "K", "M", "G", "T", "P", "E", "Z", "Y", "R", "Q", "Q+"];
    if n < 1000 {
        return n.to_string();
    }
    let mut value = n as f64;
    let mut unit = 0;
    while value >= 1000.0 && unit + 1 < UNITS.len() {
        value /= 1000.0;
        unit += 1;
    }
    if value < 10.0 {
        format!("{:.2}{}", value, UNITS[unit])
    } else if value < 100.0 {
        format!("{:.1}{}", value, UNITS[unit])
    } else {
        format!("{:.0}{}", value, UNITS[unit])
    }
}

#[cfg(test)]
mod tests {
    use super::compact;

    #[test]
    fn compact_leaves_small_counts_exact() {
        assert_eq!(compact(0), "0");
        assert_eq!(compact(999), "999");
    }

    #[test]
    fn compact_shortens_large_counts() {
        assert_eq!(compact(1_000), "1.00K");
        assert_eq!(compact(12_345), "12.3K");
        assert_eq!(compact(999_999), "1000K");
        assert_eq!(compact(1_500_000), "1.50M");
    }

    #[test]
    fn compact_saturates_at_the_widest_unit() {

        assert_eq!(compact(u128::MAX), "340282Q+");
    }
}
