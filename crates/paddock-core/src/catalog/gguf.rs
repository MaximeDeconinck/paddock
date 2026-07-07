//! Minimal GGUF header parser: just enough metadata to estimate memory/speed
//! without downloading the model. Spec: github.com/ggml-org/ggml/docs/gguf.md

use crate::PaddockError;

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
    /// `general.parameter_count` - total weight count, when published.
    pub parameter_count: Option<u64>,
    /// `{arch}.expert_count` - total experts of a MoE model.
    pub expert_count: Option<u64>,
    /// `{arch}.expert_used_count` - experts active per token.
    pub expert_used_count: Option<u64>,
    /// `{arch}.expert_feed_forward_length` - per-expert FFN hidden size (MoE).
    pub expert_feed_forward_length: Option<u64>,
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
    fn take(&mut self, n: usize) -> Result<&'a [u8], PaddockError> {
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

    fn u32(&mut self) -> Result<u32, PaddockError> {
        // Infallible: take(4) returns exactly 4 bytes or errors out first.
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64, PaddockError> {
        // Infallible: take(8) returns exactly 8 bytes or errors out first.
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn string(&mut self) -> Result<String, PaddockError> {
        let len = self.u64()?;
        if len > MAX_STRING_LEN {
            return Err(err("string length out of bounds"));
        }
        let bytes = self.take(len as usize)?;
        Ok(String::from_utf8_lossy(bytes).into_owned())
    }
}

fn err(msg: &str) -> PaddockError {
    PaddockError::Gguf(msg.to_string())
}

fn scalar_size(ty: u32) -> Result<usize, PaddockError> {
    match ty {
        0 | 1 | 7 => Ok(1), // u8, i8, bool
        2 | 3 => Ok(2),     // u16, i16
        4..=6 => Ok(4),     // u32, i32, f32
        10..=12 => Ok(8),   // u64, i64, f64
        _ => Err(err("unknown value type")),
    }
}

/// Decode a scalar value as a non-negative integer per its GGUF type.
/// Returns None when the value cannot represent a tracked count (negative
/// signed integer, non-finite or non-integer float).
fn decode_scalar_u64(ty: u32, raw: &[u8]) -> Option<u64> {
    fn float_to_u64(v: f64) -> Option<u64> {
        (v.is_finite() && v >= 0.0 && v.fract() == 0.0 && v <= u64::MAX as f64).then_some(v as u64)
    }
    fn signed_to_u64(v: i64) -> Option<u64> {
        u64::try_from(v).ok()
    }
    match ty {
        0 => Some(raw[0] as u64),                                            // u8
        1 => signed_to_u64(raw[0] as i8 as i64),                             // i8
        2 => Some(u16::from_le_bytes(raw.try_into().ok()?) as u64),          // u16
        3 => signed_to_u64(i16::from_le_bytes(raw.try_into().ok()?) as i64), // i16
        4 => Some(u32::from_le_bytes(raw.try_into().ok()?) as u64),          // u32
        5 => signed_to_u64(i32::from_le_bytes(raw.try_into().ok()?) as i64), // i32
        6 => float_to_u64(f32::from_le_bytes(raw.try_into().ok()?) as f64),  // f32
        7 => Some((raw[0] != 0) as u64),                                     // bool
        10 => Some(u64::from_le_bytes(raw.try_into().ok()?)),                // u64
        11 => signed_to_u64(i64::from_le_bytes(raw.try_into().ok()?)),       // i64
        12 => float_to_u64(f64::from_le_bytes(raw.try_into().ok()?)),        // f64
        _ => None,
    }
}

fn all_tracked_filled(m: &GgufMeta) -> bool {
    // Includes the optional-in-practice fields (parameter_count, expert_*):
    // early exit must never skip a tracked key still ahead in the buffer.
    // When a file lacks them the full (in-memory, <= a few MiB) buffer is
    // scanned instead - skipping bytes is cheap, and probe truncation is
    // handled separately in `parse_gguf_header`.
    m.architecture.is_some()
        && m.name.is_some()
        && m.block_count.is_some()
        && m.head_count.is_some()
        && m.head_count_kv.is_some()
        && m.embedding_length.is_some()
        && m.context_length.is_some()
        && m.parameter_count.is_some()
        && m.expert_count.is_some()
        && m.expert_used_count.is_some()
}

/// Parse one key/value pair, updating `meta` for tracked keys.
fn parse_kv(r: &mut Reader, meta: &mut GgufMeta) -> Result<(), PaddockError> {
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
            value_u64 = decode_scalar_u64(ty, raw);
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
    } else if key == "general.parameter_count" {
        meta.parameter_count = value_u64;
    } else if key.ends_with(".expert_used_count") {
        meta.expert_used_count = value_u64;
    } else if key.ends_with(".expert_feed_forward_length") {
        meta.expert_feed_forward_length = value_u64;
    } else if key.ends_with(".expert_count") {
        meta.expert_count = value_u64;
    }
    let _ = value_str; // keys we don't track are simply skipped
    Ok(())
}

pub fn parse_gguf_header(bytes: &[u8]) -> Result<GgufMeta, PaddockError> {
    let mut r = Reader { buf: bytes, pos: 0 };
    if r.take(4)? != b"GGUF" {
        return Err(err("bad magic, not a GGUF file"));
    }
    let version = r.u32()?;
    // v1 uses u32 counts/string lengths; parsing it with the v2/v3 layout
    // would silently misparse, so it is rejected outright.
    if !(2..=3).contains(&version) {
        return Err(err("unsupported GGUF version"));
    }
    let _tensor_count = r.u64()?;
    let kv_count = r.u64()?;
    if kv_count > MAX_KV_COUNT {
        return Err(err("kv count out of bounds"));
    }

    let mut meta = GgufMeta::default();
    for _ in 0..kv_count {
        if let Err(e) = parse_kv(&mut r, &mut meta) {
            // HTTP probes only fetch the first ~2 MiB: real headers place
            // huge tokenizer arrays after the arch keys, so a truncation
            // after useful keys were parsed still yields usable metadata.
            let truncated = matches!(&e, PaddockError::Gguf(m) if m == "truncated header");
            if truncated && (meta.architecture.is_some() || meta.block_count.is_some()) {
                return Ok(meta);
            }
            return Err(e);
        }
        if all_tracked_filled(&meta) {
            return Ok(meta); // no need to scan the (often huge) remainder
        }
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

        pub(crate) fn u64(mut self, key: &str, val: u64) -> Self {
            self.push_key(key);
            self.kvs.extend(10u32.to_le_bytes()); // type 10 = u64
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

        pub(crate) fn i32(mut self, key: &str, val: i32) -> Self {
            self.push_key(key);
            self.kvs.extend(5u32.to_le_bytes()); // type 5 = i32
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
    fn rejects_v1_header() {
        // v1 uses u32 counts/lengths; we only support v2/v3 layout.
        let mut bytes = llama_header();
        bytes[4..8].copy_from_slice(&1u32.to_le_bytes());
        assert!(parse_gguf_header(&bytes).is_err());
    }

    #[test]
    fn block_count_as_f32_decodes_as_integer() {
        let bytes = GgufBuilder::new()
            .string("general.architecture", "llama")
            .f32("llama.block_count", 32.0)
            .build();
        let m = parse_gguf_header(&bytes).unwrap();
        assert_eq!(m.block_count, Some(32));
    }

    #[test]
    fn block_count_negative_i32_is_none() {
        let bytes = GgufBuilder::new()
            .string("general.architecture", "llama")
            .i32("llama.block_count", -5)
            .build();
        let m = parse_gguf_header(&bytes).unwrap();
        assert_eq!(m.block_count, None, "negative i32 must not become huge u64");
    }

    #[test]
    fn non_integer_f32_is_none() {
        let bytes = GgufBuilder::new()
            .string("general.architecture", "llama")
            .f32("llama.block_count", 32.5)
            .build();
        let m = parse_gguf_header(&bytes).unwrap();
        assert_eq!(m.block_count, None);
    }

    #[test]
    fn truncated_header_is_error_not_panic() {
        let full = llama_header();
        // Cuts before any tracked key completes must stay Err.
        for cut in [0, 3, 4, 11, 25] {
            assert!(parse_gguf_header(&full[..cut]).is_err(), "cut at {cut}");
        }
        // A cut inside the trailing array happens AFTER architecture and
        // block_count were parsed: partial meta is returned instead of Err.
        let m = parse_gguf_header(&full[..full.len() - 1]).unwrap();
        assert_eq!(m.architecture.as_deref(), Some("llama"));
        assert_eq!(m.block_count, Some(32));
    }

    #[test]
    fn truncated_mid_giant_array_returns_partial_meta() {
        // Simulates a 2 MiB probe cutting inside a huge tokenizer array.
        let full = GgufBuilder::new()
            .string("general.architecture", "llama")
            .u32("llama.block_count", 32)
            .u32_array("tokenizer.ggml.token_ids", &vec![7u32; 100_000])
            .build();
        let cut = full.len() - 50_000; // well inside the array data
        let m = parse_gguf_header(&full[..cut]).unwrap();
        assert_eq!(m.architecture.as_deref(), Some("llama"));
        assert_eq!(m.block_count, Some(32));
    }

    #[test]
    fn early_exit_once_all_tracked_fields_filled() {
        // Every tracked field appears before a giant array; the parser must
        // return without needing the array bytes at all.
        let full = GgufBuilder::new()
            .string("general.architecture", "llama")
            .string("general.name", "Llama Test")
            .u64("general.parameter_count", 8_030_000_000)
            .u32("llama.block_count", 32)
            .u32("llama.attention.head_count", 32)
            .u32("llama.attention.head_count_kv", 8)
            .u32("llama.embedding_length", 4096)
            .u32("llama.context_length", 131072)
            .u32("llama.expert_count", 8)
            .u32("llama.expert_used_count", 2)
            .u32_array("tokenizer.ggml.token_ids", &vec![7u32; 100_000])
            .build();
        let cut = full.len() - 200_000; // cut deep inside the array
        let m = parse_gguf_header(&full[..cut]).unwrap();
        assert_eq!(m.architecture.as_deref(), Some("llama"));
        assert_eq!(m.name.as_deref(), Some("Llama Test"));
        assert_eq!(m.parameter_count, Some(8_030_000_000));
        assert_eq!(m.block_count, Some(32));
        assert_eq!(m.head_count, Some(32));
        assert_eq!(m.head_count_kv, Some(8));
        assert_eq!(m.embedding_length, Some(4096));
        assert_eq!(m.context_length, Some(131072));
        assert_eq!(m.expert_count, Some(8));
        assert_eq!(m.expert_used_count, Some(2));
    }

    #[test]
    fn parameter_count_and_expert_counts_parsed() {
        let bytes = GgufBuilder::new()
            .string("general.architecture", "qwen3moe")
            .u64("general.parameter_count", 30_500_000_000)
            .u32("qwen3moe.expert_count", 128)
            .u32("qwen3moe.expert_used_count", 8)
            .build();
        let m = parse_gguf_header(&bytes).unwrap();
        assert_eq!(m.parameter_count, Some(30_500_000_000));
        assert_eq!(m.expert_count, Some(128));
        assert_eq!(m.expert_used_count, Some(8));
        // `.expert_used_count` must not also fill `.expert_count` (or vice
        // versa) through the suffix matching.
        let dense = GgufBuilder::new()
            .string("general.architecture", "llama")
            .build();
        let m = parse_gguf_header(&dense).unwrap();
        assert_eq!(m.parameter_count, None);
        assert_eq!(m.expert_count, None);
        assert_eq!(m.expert_used_count, None);
    }

    #[test]
    fn expert_feed_forward_length_parsed() {
        let bytes = GgufBuilder::new()
            .string("general.architecture", "qwen3moe")
            .u32("qwen3moe.expert_count", 128)
            .u32("qwen3moe.expert_used_count", 8)
            .u32("qwen3moe.expert_feed_forward_length", 768)
            .build();
        let m = parse_gguf_header(&bytes).unwrap();
        assert_eq!(m.expert_feed_forward_length, Some(768));
        // The new key must not cross-populate the count fields.
        assert_eq!(m.expert_count, Some(128));
        assert_eq!(m.expert_used_count, Some(8));
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
