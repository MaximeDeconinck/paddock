/// All OS-level reads behind one trait: mockable in tests, swappable for Linux later.
pub trait SystemProbe: Send + Sync {
    fn sysctl_string(&self, name: &str) -> Option<String>;
    fn sysctl_u64(&self, name: &str) -> Option<u64>;
    /// Metal device recommendedMaxWorkingSetSize, None when no GPU/not macOS.
    fn gpu_recommended_working_set(&self) -> Option<u64>;
    /// Absolute path of a binary found in PATH.
    fn which(&self, binary: &str) -> Option<String>;
    /// Run a command, return merged stdout+stderr if exit status 0.
    fn run_command(&self, program: &str, args: &[&str]) -> Option<String>;
    /// Plain HTTP GET (used for localhost runtime probes only), body on 2xx.
    fn http_get_local(&self, url: &str) -> Option<String>;
}

pub struct RealSystemProbe;

impl SystemProbe for RealSystemProbe {
    fn sysctl_string(&self, name: &str) -> Option<String> {
        self.run_command("/usr/sbin/sysctl", &["-n", name])
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    }

    fn sysctl_u64(&self, name: &str) -> Option<u64> {
        self.sysctl_string(name)?.parse().ok()
    }

    fn gpu_recommended_working_set(&self) -> Option<u64> {
        metal_working_set()
    }

    fn which(&self, binary: &str) -> Option<String> {
        let path = std::env::var_os("PATH")?;
        for dir in std::env::split_paths(&path) {
            let candidate = dir.join(binary);
            if candidate.is_file() {
                return Some(candidate.to_string_lossy().into_owned());
            }
        }
        None
    }

    fn run_command(&self, program: &str, args: &[&str]) -> Option<String> {
        let out = std::process::Command::new(program)
            .args(args)
            .output()
            .ok()?;
        if !out.status.success() {
            return None;
        }
        let mut s = String::from_utf8_lossy(&out.stdout).into_owned();
        s.push_str(&String::from_utf8_lossy(&out.stderr));
        Some(s)
    }

    fn http_get_local(&self, url: &str) -> Option<String> {
        // Tiny HTTP/1.1 client over TcpStream: avoids pulling blocking reqwest
        // into sync hardware scan (which may run inside a tokio runtime).
        use std::io::{Read, Write};
        use std::time::Duration;
        let rest = url.strip_prefix("http://")?;
        let (host_port, path) = rest.split_once('/').unwrap_or((rest, ""));
        let addr: std::net::SocketAddr = host_port.parse().ok()?;
        let mut stream =
            std::net::TcpStream::connect_timeout(&addr, Duration::from_millis(300)).ok()?;
        stream
            .set_read_timeout(Some(Duration::from_millis(500)))
            .ok()?;
        write!(
            stream,
            "GET /{path} HTTP/1.1\r\nHost: {host_port}\r\nConnection: close\r\n\r\n"
        )
        .ok()?;
        let mut buf = String::new();
        stream.read_to_string(&mut buf).ok()?;
        if !buf.starts_with("HTTP/1.1 2") && !buf.starts_with("HTTP/1.0 2") {
            return None;
        }
        buf.split_once("\r\n\r\n").map(|(_, body)| body.to_string())
    }
}

#[cfg(target_os = "macos")]
fn metal_working_set() -> Option<u64> {
    use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice};
    // Both MTLCreateSystemDefaultDevice and recommendedMaxWorkingSetSize are
    // safe in objc2-metal 0.3: the crate exposes them as safe Rust functions
    // with no unsafe block required at the call site.
    let device = MTLCreateSystemDefaultDevice()?;
    Some(device.recommendedMaxWorkingSetSize())
}

#[cfg(not(target_os = "macos"))]
fn metal_working_set() -> Option<u64> {
    None
}

/// Test double; also useful for future `paddock scan --simulate`.
#[derive(Default)]
pub struct MockProbe {
    pub strings: std::collections::HashMap<String, String>,
    pub u64s: std::collections::HashMap<String, u64>,
    pub gpu_working_set: Option<u64>,
    pub binaries: std::collections::HashMap<String, String>,
    pub commands: std::collections::HashMap<String, String>, // key: "program arg1 arg2"
    pub http: std::collections::HashMap<String, String>,
}

impl SystemProbe for MockProbe {
    fn sysctl_string(&self, name: &str) -> Option<String> {
        self.strings.get(name).cloned()
    }
    fn sysctl_u64(&self, name: &str) -> Option<u64> {
        self.u64s.get(name).copied()
    }
    fn gpu_recommended_working_set(&self) -> Option<u64> {
        self.gpu_working_set
    }
    fn which(&self, binary: &str) -> Option<String> {
        self.binaries.get(binary).cloned()
    }
    fn run_command(&self, program: &str, args: &[&str]) -> Option<String> {
        self.commands
            .get(&format!("{program} {}", args.join(" ")))
            .cloned()
    }
    fn http_get_local(&self, url: &str) -> Option<String> {
        self.http.get(url).cloned()
    }
}
