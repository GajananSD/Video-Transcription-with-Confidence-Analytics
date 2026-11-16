"""SQLite Database Connection and Schema Management."""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

# APP_DB_PATH lets tests (or deployments) use an isolated database file.
DB_PATH = Path(
    os.getenv("APP_DB_PATH")
    or Path(__file__).resolve().parents[2] / "storage" / "app.db"
)


def get_db_connection() -> sqlite3.Connection:
    """Return a thread-safe connection to the SQLite database."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initialize SQLite database tables and indexes."""
    conn = get_db_connection()
    with conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS recordings (
                recording_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                filepath TEXT NOT NULL,
                duration REAL DEFAULT 0.0,
                size_bytes INTEGER DEFAULT 0,
                sha256 TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                recording_id TEXT NOT NULL,
                status TEXT NOT NULL,
                progress REAL DEFAULT 0.0,
                stage TEXT DEFAULT 'idle',
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (recording_id) REFERENCES recordings (recording_id)
            );

            CREATE TABLE IF NOT EXISTS transcripts (
                recording_id TEXT PRIMARY KEY,
                result_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (recording_id) REFERENCES recordings (recording_id)
            );

            CREATE TABLE IF NOT EXISTS feedback (
                feedback_id TEXT PRIMARY KEY,
                recording_id TEXT NOT NULL,
                turn_id TEXT,
                word_index INTEGER,
                original_text TEXT,
                corrected_text TEXT,
                original_speaker TEXT,
                corrected_speaker TEXT,
                notes TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (recording_id) REFERENCES recordings (recording_id)
            );

            -- One row per original word in a user-reviewed turn: the label says whether the
            -- user kept the word (1) or had to change it (0). Used to re-fit calibration.
            CREATE TABLE IF NOT EXISTS calibration_samples (
                sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
                feedback_id TEXT NOT NULL,
                recording_id TEXT NOT NULL,
                word TEXT,
                raw_confidence REAL NOT NULL,
                calibrated_confidence REAL,
                label INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (feedback_id) REFERENCES feedback (feedback_id)
            );

            -- Extensions: external services (e.g. another team's) that receive transcript
            -- events and answer with an insight card shown under the transcript.
            CREATE TABLE IF NOT EXISTS extensions (
                extension_id TEXT PRIMARY KEY,
                team TEXT NOT NULL DEFAULT '',   -- team that owns the plugin
                name TEXT NOT NULL,
                url TEXT NOT NULL,
                events TEXT NOT NULL,            -- JSON list of subscribed event names
                enabled INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL
            );

            -- Latest answer of each extension for each recording. `fingerprint` identifies
            -- the transcript version it was computed from, so edits make it stale.
            CREATE TABLE IF NOT EXISTS extension_runs (
                extension_id TEXT NOT NULL,
                recording_id TEXT NOT NULL,
                event TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                ok INTEGER NOT NULL,
                card_json TEXT,
                error TEXT,
                raw_json TEXT,                   -- exactly what the plugin answered
                elapsed_ms INTEGER,
                ran_at TEXT NOT NULL,
                PRIMARY KEY (extension_id, recording_id)
            );

            CREATE INDEX IF NOT EXISTS idx_samples_feedback ON calibration_samples(feedback_id);
            CREATE INDEX IF NOT EXISTS idx_feedback_recording ON feedback(recording_id);
            CREATE INDEX IF NOT EXISTS idx_recordings_sha256 ON recordings(sha256);
            CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
        """)
        # Columns added after the first release: add them to older databases.
        for table, column, ddl in _ADDED_COLUMNS:
            existing = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
            if column not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
    conn.close()


_ADDED_COLUMNS = [
    ("extensions", "team", "TEXT NOT NULL DEFAULT ''"),
    ("extension_runs", "raw_json", "TEXT"),
]
