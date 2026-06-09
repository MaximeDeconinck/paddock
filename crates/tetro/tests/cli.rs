use assert_cmd::Command;

fn tetro() -> Command {
    let mut c = Command::cargo_bin("tetro").unwrap();
    // isolate the catalog DB per test run
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("catalog.db");
    c.env("TETRO_DB_PATH", &path);
    // leak tempdir so it outlives the command
    std::mem::forget(dir);
    c
}

#[test]
fn scan_json_has_required_fields() {
    let out = tetro().args(["scan", "--json"]).assert().success();
    let v: serde_json::Value =
        serde_json::from_slice(&out.get_output().stdout).expect("valid json");
    assert!(v["chip_name"].is_string());
    assert!(v["ram_total_bytes"].is_u64());
    assert!(v["gpu"]["effective_limit_bytes"].is_u64());
    assert!(v["runtimes"]["ollama"]["installed"].is_boolean());
}

#[test]
fn fit_json_on_empty_catalog_is_empty_array() {
    let out = tetro().args(["fit", "--json"]).assert().success();
    let v: serde_json::Value = serde_json::from_slice(&out.get_output().stdout).unwrap();
    assert!(v.is_array());
    assert!(v.as_array().unwrap().is_empty());
}

#[test]
fn fit_on_empty_catalog_hints_sync_on_stderr() {
    tetro()
        .args(["fit", "--cli"])
        .assert()
        .success()
        .stderr(predicates::str::contains("tetro sync"));
}

#[test]
fn run_unknown_model_fails_actionably() {
    tetro()
        .args(["run", "definitely-not-a-model"])
        .assert()
        .failure()
        .stderr(predicates::str::contains("tetro sync"));
}

#[test]
fn recommend_json_is_array_max_5() {
    let out = tetro()
        .args(["recommend", "--use-case", "coding", "--json"])
        .assert()
        .success();
    let v: serde_json::Value = serde_json::from_slice(&out.get_output().stdout).unwrap();
    assert!(v.as_array().unwrap().len() <= 5);
}
