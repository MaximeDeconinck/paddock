//! Background servers refresh: a detached thread re-probes the serving state on
//! its own interval and sends a snapshot (running + available) over an mpsc
//! channel the event loop drains. Off the UI thread because the probes block.
//! Exits when the receiver is dropped (the next `send` errors).

use std::sync::mpsc::{Receiver, channel};
use std::time::Duration;

use paddock_core::hardware::RealSystemProbe;
use paddock_core::serving::{AvailableRow, History, Registry, ServerRow, list_all_servers, list_available};

const REFRESH_EVERY: Duration = Duration::from_secs(2);

pub struct ServersSnapshot {
    pub running: Vec<ServerRow>,
    pub available: Vec<AvailableRow>,
}

pub fn spawn_servers_refresh() -> Receiver<ServersSnapshot> {
    let (tx, rx) = channel();
    std::thread::spawn(move || {
        loop {
            let registry = Registry::open_default();
            let history = History::open_default();
            let running = list_all_servers(&registry, &RealSystemProbe);
            let available = list_available(&history, &RealSystemProbe, &running);
            if tx.send(ServersSnapshot { running, available }).is_err() {
                break;
            }
            std::thread::sleep(REFRESH_EVERY);
        }
    });
    rx
}
