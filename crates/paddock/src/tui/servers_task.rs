//! Background servers refresh: a detached thread re-probes the serving
//! `Registry` on its own interval and sends each snapshot over an mpsc channel
//! the event loop drains. Kept off the UI thread because `list_live` makes
//! blocking HTTP readiness probes. The thread exits when the receiver is
//! dropped (TUI quit): the next `send` returns Err and the loop breaks.

use std::sync::mpsc::{Receiver, channel};
use std::time::Duration;

use paddock_core::hardware::RealSystemProbe;
use paddock_core::serving::{Registry, ServerRow};

const REFRESH_EVERY: Duration = Duration::from_secs(2);

/// Spawn the periodic refresh. The receiver yields a fresh snapshot roughly
/// every 2s until the TUI drops it.
pub fn spawn_servers_refresh() -> Receiver<Vec<ServerRow>> {
    let (tx, rx) = channel();
    std::thread::spawn(move || {
        loop {
            let snapshot =
                paddock_core::serving::list_all_servers(&Registry::open_default(), &RealSystemProbe);
            if tx.send(snapshot).is_err() {
                break; // receiver dropped — TUI exited
            }
            std::thread::sleep(REFRESH_EVERY);
        }
    });
    rx
}
