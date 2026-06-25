use assert_cmd::Command;
use tempfile::TempDir;

/// Returns the command plus the tempdir guard that owns the isolated catalog
/// DB and serving registry; the caller binds the guard so the dir lives until
/// the test ends.
fn paddock() -> (Command, TempDir) {
    let mut c = Command::cargo_bin("paddock").unwrap();
    let dir = tempfile::tempdir().unwrap();
    c.env("PADDOCK_DB_PATH", dir.path().join("catalog.db"));
    c.env("PADDOCK_SERVING_DIR", dir.path().join("serving"));
    (c, dir)
}

/// Seed the test catalog with one small Ollama-source model so serve/run
/// planning works without network or a real sync.
fn seed_one_model(dir: &std::path::Path) {
    use paddock_core::catalog::{CatalogModel, CatalogVariant, RuntimeKind, Source, db::Db};
    let db = Db::open(dir.join("catalog.db")).unwrap();
    db.upsert_model(&CatalogModel {
        id: 0,
        name: "fake-model".into(),
        family: Some("llama".into()),
        source: Source::Ollama,
        repo: None,
        params_total: 1_000_000_000,
        params_active: 1_000_000_000,
        architecture: Some("llama".into()),
        context_max: 8192,
        released_at: None,
        released_approx: false,
        variants: vec![CatalogVariant {
            quant: "Q4_K_M".into(),
            bpw: 4.83,
            file_size_bytes: None,
            layers: 16,
            kv_heads: 8,
            head_dim: 64,
            embedding_dim: 2048,
            runtime_compat: vec![RuntimeKind::Ollama],
            source_tag: None,
        }],
    })
    .unwrap();
}

#[test]
fn scan_json_has_required_fields() {
    let (mut cmd, _dir) = paddock();
    let out = cmd.args(["scan", "--json"]).assert().success();
    let v: serde_json::Value =
        serde_json::from_slice(&out.get_output().stdout).expect("valid json");
    assert!(v["chip_name"].is_string());
    assert!(v["ram_total_bytes"].is_u64());
    assert!(v["gpu"]["effective_limit_bytes"].is_u64());
    assert!(v["runtimes"]["ollama"]["installed"].is_boolean());
}

#[test]
fn fit_json_on_empty_catalog_is_empty_array() {
    let (mut cmd, _dir) = paddock();
    let out = cmd.args(["fit", "--json"]).assert().success();
    let v: serde_json::Value = serde_json::from_slice(&out.get_output().stdout).unwrap();
    assert!(v.is_array());
    assert!(v.as_array().unwrap().is_empty());
}

#[test]
fn fit_on_empty_catalog_hints_sync_on_stderr() {
    let (mut cmd, _dir) = paddock();
    cmd.args(["fit", "--cli"])
        .assert()
        .success()
        .stderr(predicates::str::contains("paddock sync"));
}

#[test]
fn run_unknown_model_fails_actionably() {
    let (mut cmd, _dir) = paddock();
    cmd.args(["run", "definitely-not-a-model"])
        .assert()
        .failure()
        .stderr(predicates::str::contains("paddock sync"));
}

#[test]
fn serve_unknown_model_fails_actionably() {
    let (mut cmd, _dir) = paddock();
    cmd.args(["serve", "definitely-not-a-model"])
        .assert()
        .failure()
        .stderr(predicates::str::contains("paddock sync"));
}

#[test]
fn serve_json_prints_plan_without_side_effects() {
    let (mut cmd, dir) = paddock();
    seed_one_model(dir.path());
    let out = cmd
        .args(["serve", "fake-model", "--json"])
        .assert()
        .success();
    let v: serde_json::Value = serde_json::from_slice(&out.get_output().stdout).unwrap();
    assert!(
        v["endpoint"]
            .as_str()
            .unwrap()
            .starts_with("http://127.0.0.1:")
    );
    assert!(
        v["openai_url"]
            .as_str()
            .unwrap()
            .ends_with("/v1/chat/completions")
    );
    assert!(v["model_ref"].is_string());
    // zero side effects: no serving record was written
    assert!(!dir.path().join("serving").exists());
}

#[test]
fn sync_help_lists_catalog_flags() {
    let (mut cmd, _dir) = paddock();
    let out = cmd.args(["sync", "--help"]).assert().success();
    let help = String::from_utf8_lossy(&out.get_output().stdout).to_string();
    assert!(help.contains("--hf-limit"), "missing --hf-limit:\n{help}");
    assert!(help.contains("--mlx-limit"), "missing --mlx-limit:\n{help}");
    assert!(
        help.contains("--no-ollama-registry"),
        "missing --no-ollama-registry:\n{help}"
    );
    assert!(
        help.contains("--discover-limit"),
        "missing --discover-limit:\n{help}"
    );
    assert!(
        help.contains("--no-discover"),
        "missing --no-discover:\n{help}"
    );
}

#[test]
fn ps_empty_registry_reports_no_servers() {
    let (mut cmd, _dir) = paddock();
    let out = cmd.arg("ps").assert().success();
    let stdout = String::from_utf8_lossy(&out.get_output().stdout);
    assert!(stdout.contains("no servers running"), "got: {stdout}");
}

#[test]
fn ps_json_empty_is_empty_array() {
    let (mut cmd, _dir) = paddock();
    let out = cmd.args(["ps", "--json"]).assert().success();
    let v: serde_json::Value =
        serde_json::from_slice(&out.get_output().stdout).expect("valid json");
    assert_eq!(v, serde_json::json!([]));
}

#[test]
fn recommend_json_is_array_max_5() {
    let (mut cmd, _dir) = paddock();
    let out = cmd
        .args(["recommend", "--use-case", "coding", "--json"])
        .assert()
        .success();
    let v: serde_json::Value = serde_json::from_slice(&out.get_output().stdout).unwrap();
    assert!(v.as_array().unwrap().len() <= 5);
}

#[test]
fn stop_unknown_target_errors() {
    let (mut cmd, _dir) = paddock();
    cmd.args(["stop", "nope"]).assert().failure();
}
