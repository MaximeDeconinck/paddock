//! Background catalog refresh: a detached thread runs the same `catalog::sync`
//! the CLI uses, on its own tokio runtime and its own Db connection (never the
//! TUI's), reporting terminal status over an mpsc channel the event loop drains.

use std::sync::mpsc::{Receiver, channel};

use paddock_core::catalog::db::{Db, default_db_path};
use paddock_core::catalog::hf::ReqwestClient;
use paddock_core::catalog::{SyncOptions, SyncReport, sync};

/// Terminal-only in v1: `Progress` is reserved for a future per-source counter
/// (see the design's "Progress granularity"); it is never sent today.
// wired into the event loop by Task 5
#[allow(dead_code)]
pub enum SyncMsg {
    #[allow(dead_code)]
    Progress {
        source: &'static str,
        count: usize,
    },
    Done(Box<SyncReport>),
    Failed(String),
}

/// Spawn the catalog sync on a background thread. The returned receiver yields
/// exactly one terminal message (`Done` or `Failed`) then disconnects.
// wired into the event loop by Task 5
#[allow(dead_code)]
pub fn spawn_sync(opts: SyncOptions) -> Receiver<SyncMsg> {
    let (tx, rx) = channel();
    std::thread::spawn(move || {
        let result = (|| -> Result<SyncReport, String> {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|e| format!("runtime: {e}"))?;
            let db = Db::open(default_db_path()).map_err(|e| e.to_string())?;
            let http = ReqwestClient::new().map_err(|e| e.to_string())?;
            rt.block_on(sync(&http, &db, &opts))
                .map_err(|e| e.to_string())
        })();
        let _ = match result {
            Ok(report) => tx.send(SyncMsg::Done(Box::new(report))),
            Err(e) => tx.send(SyncMsg::Failed(e)),
        };
    });
    rx
}
