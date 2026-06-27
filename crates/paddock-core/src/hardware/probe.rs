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
    /// HTTP POST of a JSON body (localhost only), body on 2xx. Patient read
    /// timeout: the one caller is the Ollama warm-up, where the response
    /// only arrives once the model finished loading (can take minutes).
    fn http_post_local(&self, url: &str, json_body: &str) -> Option<String>;
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
        http_body(&buf)
    }

    fn http_post_local(&self, url: &str, json_body: &str) -> Option<String> {
        // Same tiny TcpStream client as http_get_local, but with a read
        // timeout sized for model loading: Ollama answers the warm-up POST
        // only once the weights are in memory.
        use std::io::{Read, Write};
        use std::time::Duration;
        let rest = url.strip_prefix("http://")?;
        let (host_port, path) = rest.split_once('/').unwrap_or((rest, ""));
        let addr: std::net::SocketAddr = host_port.parse().ok()?;
        let mut stream =
            std::net::TcpStream::connect_timeout(&addr, Duration::from_millis(1000)).ok()?;
        stream
            .set_read_timeout(Some(Duration::from_secs(600)))
            .ok()?;
        write!(
            stream,
            "POST /{path} HTTP/1.1\r\nHost: {host_port}\r\nContent-Type: application/json\r\n\
             Content-Length: {}\r\nConnection: close\r\n\r\n{json_body}",
            json_body.len()
        )
        .ok()?;
        let mut buf = String::new();
        stream.read_to_string(&mut buf).ok()?;
        http_body(&buf)
    }
}

/// Extract the body from a raw HTTP/1.x response, returning None on non-2xx or
/// malformed input. De-chunks when the response used `Transfer-Encoding:
/// chunked` (Ollama's `/api/tags` does), which the bodies are otherwise parsed
/// as JSON and a raw chunked body would fail.
fn http_body(raw: &str) -> Option<String> {
    if !raw.starts_with("HTTP/1.1 2") && !raw.starts_with("HTTP/1.0 2") {
        return None;
    }
    let (headers, body) = raw.split_once("\r\n\r\n")?;
    if headers.to_lowercase().contains("transfer-encoding: chunked") {
        dechunk(body)
    } else {
        Some(body.to_string())
    }
}

/// Decode an HTTP/1.1 chunked body: repeated `<hex-size>\r\n<data>\r\n` until a
/// `0`-size chunk. None on malformed framing.
fn dechunk(mut s: &str) -> Option<String> {
    let mut out = String::new();
    loop {
        let (size_line, rest) = s.split_once("\r\n")?;
        let size = usize::from_str_radix(size_line.trim(), 16).ok()?;
        if size == 0 {
            break;
        }
        out.push_str(rest.get(..size)?);
        s = rest.get(size + 2..)?; // skip the chunk data + its trailing CRLF
    }
    Some(out)
}

#[cfg(test)]
mod http_body_tests {
    use super::*;

    #[test]
    fn dechunk_decodes_chunked_body() {
        assert_eq!(dechunk("5\r\nhello\r\n6\r\n world\r\n0\r\n\r\n").unwrap(), "hello world");
    }

    #[test]
    fn http_body_dechunks_when_chunked() {
        let resp = "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\nd\r\n{\"models\":[]}\r\n0\r\n\r\n";
        assert_eq!(http_body(resp).unwrap(), "{\"models\":[]}");
    }

    #[test]
    fn http_body_passthrough_when_not_chunked() {
        let resp = "HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}";
        assert_eq!(http_body(resp).unwrap(), "{}");
    }

    #[test]
    fn http_body_none_on_non_2xx() {
        assert!(http_body("HTTP/1.1 503 Service Unavailable\r\n\r\n").is_none());
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
    /// POST fixtures (url → response body) + a log of (url, body) received,
    /// so tests can assert on what was sent.
    pub posts: std::collections::HashMap<String, String>,
    pub post_bodies: std::sync::Mutex<Vec<(String, String)>>,
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
    fn http_post_local(&self, url: &str, json_body: &str) -> Option<String> {
        self.post_bodies
            .lock()
            .unwrap()
            .push((url.to_string(), json_body.to_string()));
        self.posts.get(url).cloned()
    }
}
