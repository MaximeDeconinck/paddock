//! System clipboard write, shared by the tray and the TUI. arboard is already
//! a dependency (the tray used it inline).

/// Copy `text` to the system clipboard. Best-effort: a failure logs to stderr
/// and is otherwise ignored (no UI should crash because copy failed).
pub fn copy_to_clipboard(text: &str) {
    let res = arboard::Clipboard::new().and_then(|mut c| c.set_text(text.to_string()));
    if let Err(e) = res {
        eprintln!("could not copy to clipboard: {e}");
    }
}
