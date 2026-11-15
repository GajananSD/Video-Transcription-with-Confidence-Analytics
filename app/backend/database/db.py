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

            CREATE INDEX IF NOT EXISTS idx_samples_feedback ON calibration_samples(feedback_id);
            CREATE INDEX IF NOT EXISTS idx_feedback_recording ON feedback(recording_id);
            CREATE INDEX IF NOT EXISTS idx_recordings_sha256 ON recordings(sha256);
            CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
        """)
    conn.close()
