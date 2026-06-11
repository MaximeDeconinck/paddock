use std::path::{Path, PathBuf};

use rusqlite::{Connection, params};

use super::{CatalogModel, CatalogVariant, RuntimeKind, Source};
use crate::PaddockError;

pub struct Db {
    conn: Connection,
}

/// Default catalog location; `PADDOCK_DB_PATH` overrides (used by integration tests).
pub fn default_db_path() -> PathBuf {
    if let Ok(p) = std::env::var("PADDOCK_DB_PATH") {
        return PathBuf::from(p);
    }
    crate::paths::app_support_dir().join("catalog.db")
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
    released_at INTEGER,
    released_approx INTEGER NOT NULL DEFAULT 0,
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
    source_tag TEXT,
    UNIQUE(model_id, quant)
);
";

impl Db {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, PaddockError> {
        if let Some(parent) = path.as_ref().parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| PaddockError::Other(format!("cannot create {parent:?}: {e}")))?;
        }
        let conn = Connection::open(path)?;
        conn.execute_batch("PRAGMA foreign_keys = ON;")?;
        conn.execute_batch(SCHEMA)?;
        // Migration for DBs created before source_tag existed. SQLite has no
        // ADD COLUMN IF NOT EXISTS, so ignore the duplicate-column error.
        if let Err(e) = conn.execute("ALTER TABLE variants ADD COLUMN source_tag TEXT", []) {
            if !e.to_string().contains("duplicate column name") {
                return Err(e.into());
            }
        }
        // Migration for DBs created before model release dates existed.
        for ddl in [
            "ALTER TABLE models ADD COLUMN released_at INTEGER",
            "ALTER TABLE models ADD COLUMN released_approx INTEGER NOT NULL DEFAULT 0",
        ] {
            if let Err(e) = conn.execute(ddl, []) {
                if !e.to_string().contains("duplicate column name") {
                    return Err(e.into());
                }
            }
        }
        Ok(Self { conn })
    }

    pub fn upsert_model(&self, m: &CatalogModel) -> Result<i64, PaddockError> {
        let tx = self.conn.unchecked_transaction()?;
        tx.execute(
            "INSERT INTO models (name, family, source, repo, params_total, params_active, architecture, context_max, released_at, released_approx)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
             ON CONFLICT(source, name) DO UPDATE SET
               family=excluded.family, repo=excluded.repo,
               params_total=excluded.params_total, params_active=excluded.params_active,
               architecture=excluded.architecture, context_max=excluded.context_max,
               released_at=excluded.released_at, released_approx=excluded.released_approx",
            params![
                m.name,
                m.family,
                source_str(m.source),
                m.repo,
                m.params_total as i64,
                m.params_active as i64,
                m.architecture,
                m.context_max,
                m.released_at,
                m.released_approx as i64,
            ],
        )?;
        let id: i64 = tx.query_row(
            "SELECT id FROM models WHERE source = ?1 AND name = ?2",
            params![source_str(m.source), m.name],
            |r| r.get(0),
        )?;
        for v in &m.variants {
            let compat = serde_json::to_string(&v.runtime_compat)
                .map_err(|e| PaddockError::Other(e.to_string()))?;
            tx.execute(
                "INSERT INTO variants (model_id, quant, bpw, file_size_bytes, layers, kv_heads, head_dim, embedding_dim, runtime_compat, source_tag)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
                 ON CONFLICT(model_id, quant) DO UPDATE SET
                   bpw=excluded.bpw, file_size_bytes=excluded.file_size_bytes,
                   layers=excluded.layers, kv_heads=excluded.kv_heads,
                   head_dim=excluded.head_dim, embedding_dim=excluded.embedding_dim,
                   runtime_compat=excluded.runtime_compat, source_tag=excluded.source_tag",
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
                    v.source_tag,
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

    /// Whether a model row exists for `(source, name)`. Used by sync to
    /// preserve previously enriched rows when the registry is unreachable.
    pub fn model_exists(&self, source: Source, name: &str) -> Result<bool, PaddockError> {
        let n: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM models WHERE source = ?1 AND name = ?2",
            params![source_str(source), name],
            |r| r.get(0),
        )?;
        Ok(n > 0)
    }

    pub fn list_models(&self) -> Result<Vec<CatalogModel>, PaddockError> {
        let mut stmt = self.conn.prepare(
            "SELECT id, name, family, source, repo, params_total, params_active, architecture, context_max, released_at, released_approx
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
                    released_at: r.get(9)?,
                    released_approx: r.get::<_, i64>(10)? != 0,
                    variants: Vec::new(),
                })
            })?
            .collect::<Result<_, _>>()?;
        let mut vstmt = self.conn.prepare(
            "SELECT quant, bpw, file_size_bytes, layers, kv_heads, head_dim, embedding_dim, runtime_compat, source_tag
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
                        source_tag: r.get(8)?,
                    })
                })?
                .collect::<Result<_, _>>()?;
        }
        Ok(models)
    }

    pub fn last_sync(&self) -> Result<Option<i64>, PaddockError> {
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

    pub fn set_last_sync(&self, unix_ts: i64) -> Result<(), PaddockError> {
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
            released_at: None,
            released_approx: false,
            variants: vec![CatalogVariant {
                quant: "Q4_K_M".into(),
                bpw: 4.83,
                file_size_bytes: Some(4_920_000_000),
                layers: 32,
                kv_heads: 8,
                head_dim: 128,
                embedding_dim: 4096,
                runtime_compat: vec![RuntimeKind::Ollama],
                source_tag: None,
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
            source_tag: None,
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
    fn source_tag_roundtrips() {
        let dir = tempfile::tempdir().unwrap();
        let db = Db::open(dir.path().join("catalog.db")).unwrap();
        let mut m = sample_model();
        m.variants[0].source_tag = Some("8b-instruct-q4_K_M".into());
        db.upsert_model(&m).unwrap();
        let models = db.list_models().unwrap();
        assert_eq!(
            models[0].variants[0].source_tag.as_deref(),
            Some("8b-instruct-q4_K_M")
        );

        // Upsert back to None must clear the stored tag.
        db.upsert_model(&sample_model()).unwrap();
        let models = db.list_models().unwrap();
        assert_eq!(models[0].variants[0].source_tag, None);
    }

    #[test]
    fn open_migrates_pre_source_tag_db() {
        // Build a DB file with the OLD schema (no source_tag column), as
        // shipped before the ollama tag enrichment.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("catalog.db");
        {
            let conn = rusqlite::Connection::open(&path).unwrap();
            conn.execute_batch(
                "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                 CREATE TABLE models (
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
                 CREATE TABLE variants (
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
                 INSERT INTO models (name, family, source, repo, params_total, params_active, architecture, context_max)
                 VALUES ('llama3.1:8b', 'llama', 'ollama', NULL, 8030000000, 8030000000, 'llama', 131072);
                 INSERT INTO variants (model_id, quant, bpw, file_size_bytes, layers, kv_heads, head_dim, embedding_dim, runtime_compat)
                 VALUES (1, 'Q4_K_M', 4.83, NULL, 32, 8, 128, 4096, '[\"ollama\"]');",
            )
            .unwrap();
        }

        // Reopen via Db::open: migration adds the column, everything works.
        let db = Db::open(&path).unwrap();
        let models = db.list_models().unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].variants[0].source_tag, None);

        let mut m = sample_model();
        m.variants[0].source_tag = Some("8b".into());
        db.upsert_model(&m).unwrap();
        let models = db.list_models().unwrap();
        assert_eq!(models[0].variants[0].source_tag.as_deref(), Some("8b"));

        // Re-opening an already-migrated DB must not error (duplicate column).
        drop(db);
        let db = Db::open(&path).unwrap();
        assert_eq!(db.list_models().unwrap().len(), 1);
    }

    #[test]
    fn migration_adds_released_columns_and_roundtrips() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("old.db");
        {
            // Pre-released_at schema: models table without the new columns.
            let conn = rusqlite::Connection::open(&path).unwrap();
            conn.execute_batch(
                "CREATE TABLE models (
                    id INTEGER PRIMARY KEY, name TEXT NOT NULL, family TEXT,
                    source TEXT NOT NULL, repo TEXT,
                    params_total INTEGER NOT NULL, params_active INTEGER NOT NULL,
                    architecture TEXT, context_max INTEGER NOT NULL,
                    UNIQUE(source, name));",
            )
            .unwrap();
        }
        let db = Db::open(&path).unwrap();
        let mut m = sample_model();
        m.released_at = Some(1_743_465_600);
        m.released_approx = true;
        db.upsert_model(&m).unwrap();
        let got = &db.list_models().unwrap()[0];
        assert_eq!(got.released_at, Some(1_743_465_600));
        assert!(got.released_approx);
        // None roundtrips too.
        m.released_at = None;
        m.released_approx = false;
        db.upsert_model(&m).unwrap();
        let got = &db.list_models().unwrap()[0];
        assert_eq!(got.released_at, None);
        assert!(!got.released_approx);
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
