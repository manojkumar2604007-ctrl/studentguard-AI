"""SQLite storage for assessments."""
import json
import sqlite3
from datetime import datetime

import config


def _conn():
    c = sqlite3.connect(config.DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    with _conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT, student_name TEXT, created_at TEXT,
            predicted_status TEXT, dropout_probability REAL, risk_level TEXT,
            probabilities TEXT, observations TEXT, interventions TEXT, features TEXT)""")


def save_prediction(student_id, name, r):
    with _conn() as c:
        cur = c.execute(
            """INSERT INTO predictions (student_id, student_name, created_at,
               predicted_status, dropout_probability, risk_level, probabilities,
               observations, interventions, features) VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (student_id, name, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
             r["predicted_status"], r["dropout_probability"], r["risk_level"],
             json.dumps(r["probabilities"]), json.dumps(r["observations"]),
             json.dumps(r["interventions"]), json.dumps(r["features"])))
        pid = cur.lastrowid
        if not student_id:
            student_id = f"STU{pid:03d}"
            c.execute("UPDATE predictions SET student_id=? WHERE id=?", (student_id, pid))
        return pid


def _row(r):
    d = dict(r)
    for k in ("probabilities", "observations", "interventions", "features"):
        d[k] = json.loads(d[k])
    return d


def get_prediction(pid):
    with _conn() as c:
        r = c.execute("SELECT * FROM predictions WHERE id=?", (pid,)).fetchone()
    return _row(r) if r else None


def recent(limit=10):
    with _conn() as c:
        rows = c.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [_row(r) for r in rows]


def stats():
    with _conn() as c:
        total = c.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        risk = {r[0]: r[1] for r in c.execute(
            "SELECT risk_level, COUNT(*) FROM predictions GROUP BY risk_level")}
        status = {r[0]: r[1] for r in c.execute(
            "SELECT predicted_status, COUNT(*) FROM predictions GROUP BY predicted_status")}
        avg = c.execute("SELECT AVG(dropout_probability) FROM predictions").fetchone()[0]
    return {"total": total, "risk": risk, "status": status,
            "avg_dropout": round(avg, 1) if avg is not None else None}