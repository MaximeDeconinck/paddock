use std::path::{Path, PathBuf};

use rusqlite::{params, Connection};

use super::{CatalogModel, CatalogVariant, RuntimeKind, Source};
use crate::TetroError;

pub struct Db {
    conn: Connection,
}

/// Default catalog location; `TETRO_DB_PATH` overrides (used by integration tests).
pub fn default_db_path() -> PathBuf {
    if let Ok(p) = std::env::var("TETRO_DB_PATH") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join("Library/Application Support/tetro/catalog.db")
}

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS models (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    family TEXT,
    source TEXT NOT NULL,
    repo TEXT,
    params_total INTEGER NOT NULL,
    params_active INTEGER NOT NULL,
    architecture TEXT,
    context_max INTEGER NOT NULL,
    UNIQUE(source, name)
);
CREATE TABLE IF NOT EXISTS variants (
    id INTEGER PRIMARY KEY,
    model_id INTEGER NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    quant TEXT NOT NULL,
    bpw REAL NOT NULL,
    file_size_bytes INTEGER,
    layers INTEGER NOT NULL,
    kv_heads INTEGER NOT NULL,
    head_dim INTEGER NOT NULL,
    embedding_dim INTEGER NOT NULL,
    runtime_compat TEXT NOT NULL,
    UNIQUE(model_id, quant)
);
";

impl Db {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, TetroError> {
        if let Some(parent) = path.as_ref().parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| TetroError::Other(format!("cannot create {parent:?}: {e}")))?;
        }
        let conn = Connection::open(path)?;
        conn.execute_batch("PRAGMA foreign_keys = ON;")?;
        conn.execute_batch(SCHEMA)?;
        Ok(Self { conn })
    }

    pub fn upsert_model(&self, m: &CatalogModel) -> Result<i64, TetroError> {
        let tx = self.conn.unchecked_transaction()?;
        tx.execute(
            "INSERT INTO models (name, family, source, repo, params_total, params_active, architecture, context_max)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
             ON CONFLICT(source, name) DO UPDATE SET
               family=excluded.family, repo=excluded.repo,
               params_total=excluded.params_total, params_active=excluded.params_active,
               architecture=excluded.architecture, context_max=excluded.context_max",
            params![
                m.name,
                m.family,
                source_str(m.source),
                m.repo,
                m.params_total as i64,
                m.params_active as i64,
                m.architecture,
                m.context_max,
            ],
        )?;
        let id: i64 = tx.query_row(
            "SELECT id FROM models WHERE source = ?1 AND name = ?2",
            params![source_str(m.source), m.name],
            |r| r.get(0),
        )?;
        for v in &m.variants {
            let compat = serde_json::to_string(&v.runtime_compat)
                .map_err(|e| TetroError::Other(e.to_string()))?;
            tx.execute(
                "INSERT INTO variants (model_id, quant, bpw, file_size_bytes, layers, kv_heads, head_dim, embedding_dim, runtime_compat)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
                 ON CONFLICT(model_id, quant) DO UPDATE SET
                   bpw=excluded.bpw, file_size_bytes=excluded.file_size_bytes,
                   layers=excluded.layers, kv_heads=excluded.kv_heads,
                   head_dim=excluded.head_dim, embedding_dim=excluded.embedding_dim,
                   runtime_compat=excluded.runtime_compat",
                params![
                    id,
                    v.quant,
                    v.bpw,
                    v.file_size_bytes.map(|s| s as i64),
                    v.layers,
                    v.kv_heads,
                    v.head_dim,
                    v.embedding_dim,
                    compat,
                ],
            )?;
        }
        // Prune variants that no longer exist upstream.
        if m.variants.is_empty() {
            tx.execute("DELETE FROM variants WHERE model_id = ?1", params![id])?;
        } else {
            let placeholders = (0..m.variants.len())
                .map(|i| format!("?{}", i + 2))
                .collect::<Vec<_>>()
                .join(", ");
            let sql = format!(
                "DELETE FROM variants WHERE model_id = ?1 AND quant NOT IN ({placeholders})"
            );
            let mut args: Vec<&dyn rusqlite::ToSql> = vec![&id];
            for v in &m.variants {
                args.push(&v.quant);
            }
            tx.execute(&sql, args.as_slice())?;
        }
        tx.commit()?;
        Ok(id)
    }

    pub fn list_models(&self) -> Result<Vec<CatalogModel>, TetroError> {
        let mut stmt = self.conn.prepare(
            "SELECT id, name, family, source, repo, params_total, params_active, architecture, context_max
             FROM models ORDER BY params_total DESC",
        )?;
        let mut models: Vec<CatalogModel> = stmt
            .query_map([], |r| {
                Ok(CatalogModel {
                    id: r.get(0)?,
                    name: r.get(1)?,
                    family: r.get(2)?,
                    source: parse_source(&r.get::<_, String>(3)?),
                    repo: r.get(4)?,
                    params_total: r.get::<_, i64>(5)? as u64,
                    params_active: r.get::<_, i64>(6)? as u64,
                    architecture: r.get(7)?,
                    context_max: r.get(8)?,
                    variants: Vec::new(),
                })
            })?
            .collect::<Result<_, _>>()?;
        let mut vstmt = self.conn.prepare(
            "SELECT quant, bpw, file_size_bytes, layers, kv_heads, head_dim, embedding_dim, runtime_compat
             FROM variants WHERE model_id = ?1",
        )?;
        for m in &mut models {
            m.variants = vstmt
                .query_map([m.id], |r| {
                    Ok(CatalogVariant {
                        quant: r.get(0)?,
                        bpw: r.get(1)?,
                        file_size_bytes: r.get::<_, Option<i64>>(2)?.map(|s| s as u64),
                        layers: r.get(3)?,
                        kv_heads: r.get(4)?,
                        head_dim: r.get(5)?,
                        embedding_dim: r.get(6)?,
                        runtime_compat: serde_json::from_str::<Vec<RuntimeKind>>(
                            &r.get::<_, String>(7)?,
                        )
                        .unwrap_or_default(),
                    })
                })?
                .collect::<Result<_, _>>()?;
        }
        Ok(models)
    }

    pub fn last_sync(&self) -> Result<Option<i64>, TetroError> {
        match self
            .conn
            .query_row("SELECT value FROM meta WHERE key = 'last_sync'", [], |r| {
                r.get::<_, String>(0)
            }) {
            Ok(v) => Ok(v.parse().ok()),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    pub fn set_last_sync(&self, unix_ts: i64) -> Result<(), TetroError> {
        self.conn.execute(
            "INSERT INTO meta (key, value) VALUES ('last_sync', ?1)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            params![unix_ts.to_string()],
        )?;
        Ok(())
    }
}

fn source_str(s: Source) -> &'static str {
    match s {
        Source::HuggingFace => "huggingface",
        Source::Ollama => "ollama",
        Source::Mlx => "mlx",
    }
}

fn parse_source(s: &str) -> Source {
    match s {
        "huggingface" => Source::HuggingFace,
        "mlx" => Source::Mlx,
        _ => Source::Ollama,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::{CatalogModel, CatalogVariant, RuntimeKind, Source};

    fn sample_model() -> CatalogModel {
        CatalogModel {
            id: 0,
            name: "llama3.1:8b".into(),
            family: Some("llama".into()),
            source: Source::Ollama,
            repo: None,
            params_total: 8_030_000_000,
            params_active: 8_030_000_000,
            architecture: Some("llama".into()),
            context_max: 131_072,
            variants: vec![CatalogVariant {
                quant: "Q4_K_M".into(),
                bpw: 4.83,
                file_size_bytes: Some(4_920_000_000),
                layers: 32,
                kv_heads: 8,
                head_dim: 128,
                embedding_dim: 4096,
                runtime_compat: vec![RuntimeKind::Ollama],
            }],
        }
    }

    #[test]
    fn roundtrip_upsert_and_list() {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(dir.path().join("catalog.db")).unwrap();
        db.upsert_model(&sample_model()).unwrap();
        db.upsert_model(&sample_model()).unwrap(); // idempotent
        let models = db.list_models().unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].variants.len(), 1);
        assert_eq!(models[0].variants[0].quant, "Q4_K_M");
    }

    #[test]
    fn stale_variants_pruned_on_upsert() {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(dir.path().join("catalog.db")).unwrap();
        let mut m = sample_model();
        m.variants.push(CatalogVariant {
            quant: "Q8_0".into(),
            bpw: 8.5,
            file_size_bytes: Some(8_540_000_000),
            layers: 32,
            kv_heads: 8,
            head_dim: 128,
            embedding_dim: 4096,
            runtime_compat: vec![RuntimeKind::Ollama],
        });
        db.upsert_model(&m).unwrap();
        let models = db.list_models().unwrap();
        assert_eq!(models[0].variants.len(), 2);

        // Re-upsert with only Q4_K_M: Q8_0 must be pruned.
        db.upsert_model(&sample_model()).unwrap();
        let models = db.list_models().unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].variants.len(), 1);
        assert_eq!(models[0].variants[0].quant, "Q4_K_M");
    }

    #[test]
    fn meta_last_sync() {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(dir.path().join("catalog.db")).unwrap();
        assert!(db.last_sync().unwrap().is_none());
        db.set_last_sync(1_770_000_000).unwrap();
        assert_eq!(db.last_sync().unwrap(), Some(1_770_000_000));
    }
}
