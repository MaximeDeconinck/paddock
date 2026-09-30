//! `paddock mcp` over real stdio: spawn the binary, write JSON-RPC lines,
//! read the responses back. Responses can arrive out of order (tools run on
//! a thread pool), so everything is matched by request id.

use std::collections::{HashMap, HashSet};
use std::io::{BufRead, BufReader, Read, Write};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use tempfile::TempDir;

/// How long a session may take to answer every request before the test fails.
const SESSION_TIMEOUT: Duration = Duration::from_secs(30);

/// Returns the command plus the tempdir guard that owns the isolated catalog
/// DB, serving registry and calibration file.
fn paddock() -> (Command, TempDir) {
    let mut c = Command::new(assert_cmd::cargo::cargo_bin("paddock"));
    let dir = tempfile::tempdir().unwrap();
    c.env("PADDOCK_DB_PATH", dir.path().join("catalog.db"));
    c.env("PADDOCK_SERVING_DIR", dir.path().join("serving"));
    c.env(
        "PADDOCK_CALIBRATION_PATH",
        dir.path().join("calibration.json"),
    );
    (c, dir)
}

/// Seed the test catalog with one small Ollama-source model.
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

fn initialize() -> Vec<Value> {
    vec![
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{
            "protocolVersion":"2025-06-18","capabilities":{},
            "clientInfo":{"name":"paddock-test","version":"0"}}}),
        json!({"jsonrpc":"2.0","method":"notifications/initialized"}),
    ]
}

fn call(id: u64, tool: &str, args: Value) -> Value {
    json!({"jsonrpc":"2.0","id":id,"method":"tools/call","params":{"name":tool,"arguments":args}})
}

/// Run one session: `initialize` + `initialized` + `requests`. Stdin stays
/// open until every request id has a response (so no reply can be lost to an
/// early EOF), then it is closed and the server must exit cleanly. Every
/// stdout line must be JSON: stdout is the protocol channel. Returns
/// responses keyed by id.
fn session(cmd: &mut Command, requests: Vec<Value>) -> HashMap<u64, Value> {
    let mut messages = initialize();
    messages.extend(requests);
    let mut pending: HashSet<u64> = messages.iter().filter_map(|m| m["id"].as_u64()).collect();

    let mut child = cmd
        .arg("mcp")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn paddock mcp");

    let stdout = child.stdout.take().unwrap();
    let (tx, rx) = mpsc::channel::<String>();
    let out_reader = thread::spawn(move || {
        for line in BufReader::new(stdout).lines() {
            let Ok(line) = line else { break };
            if tx.send(line).is_err() {
                break;
            }
        }
    });
    let mut stderr = child.stderr.take().unwrap();
    let err_reader = thread::spawn(move || {
        let mut s = String::new();
        let _ = stderr.read_to_string(&mut s);
        s
    });

    let mut stdin = child.stdin.take().unwrap();
    for m in &messages {
        writeln!(stdin, "{m}").expect("write to paddock mcp stdin");
    }
    stdin.flush().unwrap();

    let mut by_id = HashMap::new();
    let mut received: Vec<String> = Vec::new();
    let deadline = Instant::now() + SESSION_TIMEOUT;
    while !pending.is_empty() {
        let left = deadline.saturating_duration_since(Instant::now());
        match rx.recv_timeout(left) {
            Ok(line) => {
                if line.trim().is_empty() {
                    continue;
                }
                let v: Value = serde_json::from_str(&line)
                    .unwrap_or_else(|e| panic!("stdout line is not JSON-RPC ({e}): {line}"));
                received.push(line);
                if let Some(id) = v["id"].as_u64() {
                    pending.remove(&id);
                    by_id.insert(id, v);
                }
            }
            Err(_) => {
                let _ = child.kill();
                drop(stdin);
                let stderr = err_reader.join().unwrap_or_default();
                panic!(
                    "no response for ids {pending:?} within {SESSION_TIMEOUT:?}\nreceived:\n{}\nstderr:\n{stderr}",
                    received.join("\n")
                );
            }
        }
    }

    drop(stdin);
    let status = child.wait().expect("wait for paddock mcp");
    out_reader.join().unwrap();
    // Anything printed after the last response must still be JSON-RPC.
    for line in rx.try_iter().filter(|l| !l.trim().is_empty()) {
        serde_json::from_str::<Value>(&line)
            .unwrap_or_else(|e| panic!("stdout line is not JSON-RPC ({e}): {line}"));
    }
    let stderr = err_reader.join().unwrap();
    assert!(
        status.success(),
        "paddock mcp exited with {status}\nstderr:\n{stderr}"
    );
    by_id
}

fn structured(resp: &Value) -> &Value {
    &resp["result"]["structuredContent"]
}

#[test]
fn initialize_advertises_tools_and_instructions() {
    let (mut cmd, _dir) = paddock();
    let r = session(&mut cmd, vec![]);
    let init = &r[&1]["result"];
    assert_eq!(init["serverInfo"]["name"], "paddock");
    assert!(init["capabilities"]["tools"].is_object());
    let instructions = init["instructions"].as_str().unwrap();
    assert!(instructions.contains("paddock_ps"));
    assert!(instructions.contains("no_runtime"));
}

#[test]
fn tools_list_has_six_tools_with_object_schemas() {
    let (mut cmd, _dir) = paddock();
    let r = session(
        &mut cmd,
        vec![json!({"jsonrpc":"2.0","id":2,"method":"tools/list"})],
    );
    let tools = r[&2]["result"]["tools"].as_array().unwrap();
    let mut names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
    names.sort();
    assert_eq!(
        names,
        [
            "paddock_fit",
            "paddock_ps",
            "paddock_recommend",
            "paddock_scan",
            "paddock_serve",
            "paddock_stop"
        ]
    );
    for t in tools {
        assert_eq!(t["inputSchema"]["type"], "object", "{}", t["name"]);
        assert!(t["inputSchema"]["properties"].is_object(), "{}", t["name"]);
    }
    let serve = tools.iter().find(|t| t["name"] == "paddock_serve").unwrap();
    assert_eq!(serve["inputSchema"]["required"], json!(["model"]));
    let stop = tools.iter().find(|t| t["name"] == "paddock_stop").unwrap();
    assert_eq!(stop["annotations"]["destructiveHint"], true);
}

#[test]
fn scan_returns_hardware_profile() {
    let (mut cmd, _dir) = paddock();
    let r = session(&mut cmd, vec![call(2, "paddock_scan", json!({}))]);
    let v = structured(&r[&2]);
    assert!(v["chip_name"].is_string());
    assert!(v["ram_total_bytes"].is_u64());
    assert!(v["runtimes"]["ollama"]["installed"].is_boolean());
    // The text block carries the same JSON.
    let text = r[&2]["result"]["content"][0]["text"].as_str().unwrap();
    let parsed: Value = serde_json::from_str(text).unwrap();
    assert_eq!(&parsed, v);
}

#[test]
fn ps_on_empty_registry_has_running_and_available() {
    let (mut cmd, _dir) = paddock();
    let r = session(&mut cmd, vec![call(2, "paddock_ps", json!({}))]);
    let v = structured(&r[&2]);
    assert_eq!(v["running"], json!([]));
    // `available` reflects the host's real Ollama install: shape only.
    assert!(v["available"].is_array());
}

#[test]
fn fit_on_empty_catalog_is_empty_array() {
    let (mut cmd, _dir) = paddock();
    let r = session(
        &mut cmd,
        vec![call(2, "paddock_fit", json!({"use_case": "coding"}))],
    );
    assert_eq!(structured(&r[&2])["models"], json!([]));
}

#[test]
fn fit_limit_zero_is_empty_array() {
    // Review Focus 2: a zero limit is a valid "give me nothing", not an error.
    let (mut cmd, dir) = paddock();
    seed_one_model(dir.path());
    let r = session(&mut cmd, vec![call(2, "paddock_fit", json!({"limit": 0}))]);
    assert_eq!(r[&2]["result"]["isError"], false);
    assert_eq!(structured(&r[&2])["models"], json!([]));
}

#[test]
fn fit_and_recommend_list_seeded_model() {
    let (mut cmd, dir) = paddock();
    seed_one_model(dir.path());
    let r = session(
        &mut cmd,
        vec![
            call(2, "paddock_fit", json!({})),
            call(3, "paddock_recommend", json!({})),
        ],
    );
    assert_eq!(structured(&r[&2])["models"][0]["name"], "fake-model");
    let rec = &structured(&r[&3])["recommendations"];
    assert_eq!(rec[0]["model"], "fake-model");
    assert!(rec[0]["justification"].as_str().unwrap().contains("tok/s"));
}

#[test]
fn stop_all_is_rejected_before_touching_registry() {
    let (mut cmd, _dir) = paddock();
    let r = session(
        &mut cmd,
        vec![call(2, "paddock_stop", json!({"target": "all"}))],
    );
    assert_eq!(r[&2]["result"]["isError"], true);
    assert_eq!(structured(&r[&2])["code"], "invalid_target");
}

#[test]
fn stop_unknown_pid_is_no_server_match() {
    // Review Focus 3: a numeric target that is not running.
    let (mut cmd, _dir) = paddock();
    let r = session(
        &mut cmd,
        vec![call(2, "paddock_stop", json!({"target": "99999999"}))],
    );
    assert_eq!(r[&2]["result"]["isError"], true);
    let v = structured(&r[&2]);
    assert_eq!(v["code"], "no_server_match");
    assert_eq!(v["running"], json!([]));
}

#[test]
fn serve_on_empty_catalog_is_catalog_empty() {
    let (mut cmd, _dir) = paddock();
    let r = session(
        &mut cmd,
        vec![call(2, "paddock_serve", json!({"model": "x"}))],
    );
    assert_eq!(r[&2]["result"]["isError"], true);
    assert_eq!(structured(&r[&2])["code"], "catalog_empty");
}

#[test]
fn serve_unknown_model_is_model_not_found() {
    let (mut cmd, dir) = paddock();
    seed_one_model(dir.path());
    let r = session(
        &mut cmd,
        vec![call(2, "paddock_serve", json!({"model": "nope"}))],
    );
    let v = structured(&r[&2]);
    assert_eq!(v["code"], "model_not_found");
    assert_eq!(v["hint"], "call paddock_fit for names");
    // Zero side effects: nothing was spawned or recorded.
    assert!(!dir.path().join("serving").exists());
}

#[test]
fn serve_unknown_quant_lists_available() {
    let (mut cmd, dir) = paddock();
    seed_one_model(dir.path());
    let r = session(
        &mut cmd,
        vec![call(
            2,
            "paddock_serve",
            json!({"model": "fake-model", "quant": "Q9_X"}),
        )],
    );
    let v = structured(&r[&2]);
    assert_eq!(v["code"], "unknown_quant");
    assert_eq!(v["available"], json!(["Q4_K_M"]));
}

#[test]
fn unknown_tool_does_not_kill_session() {
    // Review Focus 5: a bad call gets a JSON-RPC error; the next call still works.
    let (mut cmd, _dir) = paddock();
    let r = session(
        &mut cmd,
        vec![
            call(2, "paddock_nope", json!({})),
            call(3, "paddock_stop", json!({})), // missing required `target`
            call(4, "paddock_ps", json!({})),
        ],
    );
    assert!(
        r[&2].get("error").is_some() || r[&2]["result"]["isError"] == true,
        "{}",
        r[&2]
    );
    assert!(
        r[&3].get("error").is_some() || r[&3]["result"]["isError"] == true,
        "{}",
        r[&3]
    );
    assert_eq!(structured(&r[&4])["running"], json!([]));
}
