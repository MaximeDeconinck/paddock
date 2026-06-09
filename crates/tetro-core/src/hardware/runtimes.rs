use serde::{Deserialize, Serialize};

use super::probe::SystemProbe;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RuntimeStatus {
    pub installed: bool,
    pub version: Option<String>,
    pub running: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RuntimesStatus {
    pub ollama: RuntimeStatus,
    pub llama_cpp: RuntimeStatus,
    pub mlx: RuntimeStatus,
}

const OLLAMA_VERSION_URL: &str = "http://127.0.0.1:11434/api/version";

pub fn detect_runtimes(probe: &dyn SystemProbe) -> RuntimesStatus {
    RuntimesStatus {
        ollama: detect_ollama(probe),
        llama_cpp: detect_llama_cpp(probe),
        mlx: detect_mlx(probe),
    }
}

fn detect_ollama(probe: &dyn SystemProbe) -> RuntimeStatus {
    let in_path = probe.which("ollama").is_some();
    if let Some(body) = probe.http_get_local(OLLAMA_VERSION_URL) {
        let version = serde_json::from_str::<serde_json::Value>(&body)
            .ok()
            .and_then(|v| v["version"].as_str().map(String::from));
        // Server answering counts as installed even if the binary is not in PATH.
        return RuntimeStatus {
            installed: true,
            version,
            running: true,
        };
    }
    let version = in_path
        .then(|| probe.run_command("ollama", &["--version"]))
        .flatten()
        .and_then(|out| out.split_whitespace().last().map(String::from));
    RuntimeStatus {
        installed: in_path,
        version,
        running: false,
    }
}

fn detect_llama_cpp(probe: &dyn SystemProbe) -> RuntimeStatus {
    let bin = ["llama-cli", "llama-server"]
        .iter()
        .find(|b| probe.which(b).is_some());
    let version = bin
        .and_then(|b| probe.run_command(b, &["--version"]))
        .and_then(|out| {
            // llama.cpp prints e.g. "version: 4521 (commit)" on stderr.
            out.lines()
                .find(|l| l.contains("version"))
                .map(|l| l.trim().to_string())
        });
    RuntimeStatus {
        installed: bin.is_some(),
        version,
        running: false,
    }
}

fn detect_mlx(probe: &dyn SystemProbe) -> RuntimeStatus {
    let via_path = probe.which("mlx_lm.chat").is_some();
    let version = probe
        .run_command(
            "python3",
            &["-c", "import mlx_lm; print(mlx_lm.__version__)"],
        )
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    RuntimeStatus {
        installed: via_path || version.is_some(),
        version,
        running: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::probe::MockProbe;

    #[test]
    fn ollama_running_server_wins() {
        let mut p = MockProbe::default();
        p.binaries
            .insert("ollama".into(), "/opt/homebrew/bin/ollama".into());
        p.http.insert(
            "http://127.0.0.1:11434/api/version".into(),
            r#"{"version":"0.9.1"}"#.into(),
        );
        let s = detect_runtimes(&p);
        assert!(s.ollama.installed);
        assert!(s.ollama.running);
        assert_eq!(s.ollama.version.as_deref(), Some("0.9.1"));
    }

    #[test]
    fn ollama_installed_not_running() {
        let mut p = MockProbe::default();
        p.binaries
            .insert("ollama".into(), "/opt/homebrew/bin/ollama".into());
        p.commands.insert(
            "ollama --version".into(),
            "ollama version is 0.9.1\n".into(),
        );
        let s = detect_runtimes(&p);
        assert!(s.ollama.installed);
        assert!(!s.ollama.running);
        assert_eq!(s.ollama.version.as_deref(), Some("0.9.1"));
    }

    #[test]
    fn nothing_installed() {
        let p = MockProbe::default();
        let s = detect_runtimes(&p);
        assert!(!s.ollama.installed && !s.llama_cpp.installed && !s.mlx.installed);
    }

    #[test]
    fn mlx_via_python_import() {
        let mut p = MockProbe::default();
        p.commands.insert(
            "python3 -c import mlx_lm; print(mlx_lm.__version__)".into(),
            "0.24.0\n".into(),
        );
        let s = detect_runtimes(&p);
        assert!(s.mlx.installed);
        assert_eq!(s.mlx.version.as_deref(), Some("0.24.0"));
    }
}
