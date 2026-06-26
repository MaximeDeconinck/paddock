//! Application data directory resolution, including the one-time migration
//! from the legacy pre-rename directory (the project used to be called
//! "tetro"; pre-release, so no back-compat aliases - just a silent move).

use std::path::{Path, PathBuf};

/// Pre-rename data directory name. Sole intentional occurrence of the old
/// project name in the source tree (historical planning docs keep it);
/// required to find existing installs.
const LEGACY_DIR_NAME: &str = "tetro";

/// `~/Library/Application Support/paddock`, migrating a legacy directory in
/// place the first time we look.
pub fn app_support_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    resolve(&PathBuf::from(home).join("Library/Application Support"))
}

/// If the paddock dir is absent and the legacy dir exists, rename the whole
/// legacy dir (catalog.db + serving/) to paddock. Best-effort and one-shot:
/// errors are ignored silently - worst case the user gets a fresh dir and
/// re-syncs. If both dirs ever coexist (a past failed rename, or an old
/// binary recreating the legacy dir), the legacy dir is left orphaned and is
/// NOT cleaned up or reported - acceptable pre-1.0; manual cleanup: delete
/// ~/Library/Application Support/tetro.
fn resolve(base: &Path) -> PathBuf {
    let paddock = base.join("paddock");
    let legacy = base.join(LEGACY_DIR_NAME);
    if !paddock.exists() && legacy.exists() {
        // Benign TOCTOU: POSIX rename replaces an empty target dir, so a concurrently created empty paddock/ may be swapped for the legacy dir during startup - legacy data wins, concurrent migrations are safe (loser gets ENOENT, ignored).
        let _ = std::fs::rename(&legacy, &paddock);
    }
    paddock
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_dir_is_renamed_wholesale() {
        let tmp = tempfile::tempdir().unwrap();
        let legacy = tmp.path().join(LEGACY_DIR_NAME);
        std::fs::create_dir_all(legacy.join("serving")).unwrap();
        std::fs::write(legacy.join("catalog.db"), b"sentinel").unwrap();

        let dir = resolve(tmp.path());

        assert_eq!(dir, tmp.path().join("paddock"));
        assert!(!legacy.exists(), "legacy dir must be gone");
        assert_eq!(std::fs::read(dir.join("catalog.db")).unwrap(), b"sentinel");
        assert!(dir.join("serving").is_dir());
    }

    #[test]
    fn existing_paddock_dir_wins_over_legacy() {
        let tmp = tempfile::tempdir().unwrap();
        let legacy = tmp.path().join(LEGACY_DIR_NAME);
        std::fs::create_dir_all(&legacy).unwrap();
        std::fs::write(legacy.join("catalog.db"), b"old").unwrap();
        let paddock = tmp.path().join("paddock");
        std::fs::create_dir_all(&paddock).unwrap();
        std::fs::write(paddock.join("catalog.db"), b"new").unwrap();

        let dir = resolve(tmp.path());

        assert_eq!(std::fs::read(dir.join("catalog.db")).unwrap(), b"new");
        assert!(legacy.exists(), "legacy dir is left untouched");
    }

    #[test]
    fn fresh_install_returns_paddock_without_creating_it() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = resolve(tmp.path());
        assert_eq!(dir, tmp.path().join("paddock"));
        assert!(!dir.exists(), "resolution must not create the dir");
    }
}
