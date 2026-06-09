//! Minimal GGUF header parser: just enough metadata to estimate memory/speed
//! without downloading the model. Spec: github.com/ggml-org/ggml/docs/gguf.md

use crate::TetroError;

const MAX_KV_COUNT: u64 = 65_536;
const MAX_STRING_LEN: u64 = 1 << 20;
const MAX_ARRAY_LEN: u64 = 1 << 24;

#[derive(Debug, Default, Clone)]
pub struct GgufMeta {
    pub architecture: Option<String>,
    pub name: Option<String>,
    pub block_count: Option<u64>,
    pub head_count: Option<u64>,
    pub head_count_kv: Option<u64>,
    pub embedding_length: Option<u64>,
    pub context_length: Option<u64>,
}

impl GgufMeta {
    pub fn head_dim(&self) -> Option<u64> {
        match (self.embedding_length, self.head_count) {
            (Some(e), Some(h)) if h > 0 => Some(e / h),
            _ => None,
        }
    }
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], TetroError> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| err("offset overflow"))?;
        let s = self
            .buf
            .get(self.pos..end)
            .ok_or_else(|| err("truncated header"))?;
        self.pos = end;
        Ok(s)
    }

    fn u32(&mut self) -> Result<u32, TetroError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64, TetroError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn string(&mut self) -> Result<String, TetroError> {
        let len = self.u64()?;
        if len > MAX_STRING_LEN {
            return Err(err("string length out of bounds"));
        }
        let bytes = self.take(len as usize)?;
        Ok(String::from_utf8_lossy(bytes).into_owned())
    }
}

fn err(msg: &str) -> TetroError {
    TetroError::Gguf(msg.to_string())
}

fn scalar_size(ty: u32) -> Result<usize, TetroError> {
    match ty {
        0 | 1 | 7 => Ok(1), // u8, i8, bool
        2 | 3 => Ok(2),     // u16, i16
        4..=6 => Ok(4),     // u32, i32, f32
        10..=12 => Ok(8),   // u64, i64, f64
        _ => Err(err("unknown value type")),
    }
}

pub fn parse_gguf_header(bytes: &[u8]) -> Result<GgufMeta, TetroError> {
    let mut r = Reader { buf: bytes, pos: 0 };
    if r.take(4)? != b"GGUF" {
        return Err(err("bad magic, not a GGUF file"));
    }
    let version = r.u32()?;
    if !(1..=3).contains(&version) {
        return Err(err("unsupported GGUF version"));
    }
    let _tensor_count = r.u64()?;
    let kv_count = r.u64()?;
    if kv_count > MAX_KV_COUNT {
        return Err(err("kv count out of bounds"));
    }

    let mut meta = GgufMeta::default();
    for _ in 0..kv_count {
        let key = r.string()?;
        let ty = r.u32()?;
        let value_u64: Option<u64>;
        let value_str: Option<String>;
        match ty {
            8 => {
                value_str = Some(r.string()?);
                value_u64 = None;
            }
            9 => {
                let elem_ty = r.u32()?;
                let len = r.u64()?;
                if len > MAX_ARRAY_LEN {
                    return Err(err("array length out of bounds"));
                }
                if elem_ty == 8 {
                    for _ in 0..len {
                        r.string()?;
                    }
                } else {
                    let sz = scalar_size(elem_ty)?;
                    r.take(
                        (len as usize)
                            .checked_mul(sz)
                            .ok_or_else(|| err("array size overflow"))?,
                    )?;
                }
                value_str = None;
                value_u64 = None;
            }
            _ => {
                let sz = scalar_size(ty)?;
                let raw = r.take(sz)?;
                let mut padded = [0u8; 8];
                padded[..sz].copy_from_slice(raw);
                value_u64 = Some(u64::from_le_bytes(padded));
                value_str = None;
            }
        }

        if key == "general.architecture" {
            meta.architecture = value_str.clone();
        } else if key == "general.name" {
            meta.name = value_str.clone();
        } else if key.ends_with(".block_count") {
            meta.block_count = value_u64;
        } else if key.ends_with(".attention.head_count") {
            meta.head_count = value_u64;
        } else if key.ends_with(".attention.head_count_kv") {
            meta.head_count_kv = value_u64;
        } else if key.ends_with(".embedding_length") {
            meta.embedding_length = value_u64;
        } else if key.ends_with(".context_length") {
            meta.context_length = value_u64;
        }
        let _ = value_str; // keys we don't track are simply skipped
    }
    Ok(meta)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Minimal GGUF v3 header builder (spec: github.com/ggml-org/ggml/docs/gguf.md).
    pub(crate) struct GgufBuilder {
        kvs: Vec<u8>,
        count: u64,
    }

    impl GgufBuilder {
        pub(crate) fn new() -> Self {
            Self {
                kvs: Vec::new(),
                count: 0,
            }
        }

        fn push_key(&mut self, key: &str) {
            self.kvs.extend((key.len() as u64).to_le_bytes());
            self.kvs.extend(key.as_bytes());
        }

        pub(crate) fn string(mut self, key: &str, val: &str) -> Self {
            self.push_key(key);
            self.kvs.extend(8u32.to_le_bytes()); // type 8 = string
            self.kvs.extend((val.len() as u64).to_le_bytes());
            self.kvs.extend(val.as_bytes());
            self.count += 1;
            self
        }

        pub(crate) fn u32(mut self, key: &str, val: u32) -> Self {
            self.push_key(key);
            self.kvs.extend(4u32.to_le_bytes()); // type 4 = u32
            self.kvs.extend(val.to_le_bytes());
            self.count += 1;
            self
        }

        pub(crate) fn f32(mut self, key: &str, val: f32) -> Self {
            self.push_key(key);
            self.kvs.extend(6u32.to_le_bytes()); // type 6 = f32
            self.kvs.extend(val.to_le_bytes());
            self.count += 1;
            self
        }

        pub(crate) fn u32_array(mut self, key: &str, vals: &[u32]) -> Self {
            self.push_key(key);
            self.kvs.extend(9u32.to_le_bytes()); // type 9 = array
            self.kvs.extend(4u32.to_le_bytes()); // element type u32
            self.kvs.extend((vals.len() as u64).to_le_bytes());
            for v in vals {
                self.kvs.extend(v.to_le_bytes());
            }
            self.count += 1;
            self
        }

        pub(crate) fn build(self) -> Vec<u8> {
            let mut out = Vec::new();
            out.extend(b"GGUF");
            out.extend(3u32.to_le_bytes()); // version
            out.extend(0u64.to_le_bytes()); // tensor_count
            out.extend(self.count.to_le_bytes()); // kv count
            out.extend(self.kvs);
            out
        }
    }

    pub(crate) fn llama_header() -> Vec<u8> {
        GgufBuilder::new()
            .string("general.architecture", "llama")
            .u32("llama.block_count", 32)
            .u32("llama.attention.head_count", 32)
            .u32("llama.attention.head_count_kv", 8)
            .u32("llama.embedding_length", 4096)
            .u32("llama.context_length", 131072)
            .f32("llama.rope.freq_base", 500000.0) // irrelevant key, must be skipped
            .u32_array("llama.dummy.array", &[1, 2, 3]) // arrays must be skippable
            .build()
    }

    #[test]
    fn parses_llama_header() {
        let m = parse_gguf_header(&llama_header()).unwrap();
        assert_eq!(m.architecture.as_deref(), Some("llama"));
        assert_eq!(m.block_count, Some(32));
        assert_eq!(m.head_count, Some(32));
        assert_eq!(m.head_count_kv, Some(8));
        assert_eq!(m.embedding_length, Some(4096));
        assert_eq!(m.context_length, Some(131072));
        assert_eq!(m.head_dim(), Some(128)); // 4096 / 32
    }

    #[test]
    fn rejects_bad_magic() {
        assert!(parse_gguf_header(b"NOPE\x03\x00\x00\x00").is_err());
    }

    #[test]
    fn truncated_header_is_error_not_panic() {
        let full = llama_header();
        for cut in [0, 3, 4, 11, 25, full.len() - 1] {
            assert!(parse_gguf_header(&full[..cut]).is_err(), "cut at {cut}");
        }
    }

    #[test]
    fn absurd_lengths_rejected() {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend(3u32.to_le_bytes());
        bytes.extend(0u64.to_le_bytes());
        bytes.extend(u64::MAX.to_le_bytes()); // absurd kv count
        assert!(parse_gguf_header(&bytes).is_err());
    }
}
