# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Incremental local vector index using the original Qwen3-VL video embedding method."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import numpy as np

from .catalog import in_folder


def _unit(vector) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(value)
    if not value.size or not np.isfinite(value).all() or norm == 0:
        raise ValueError("Embedding must be a finite, nonzero vector")
    return value / norm


class VLIndex:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            db.execute("""CREATE TABLE IF NOT EXISTS vectors (
                asset_id TEXT PRIMARY KEY, category TEXT NOT NULL, video TEXT NOT NULL,
                size INTEGER NOT NULL, mtime_ns INTEGER NOT NULL, description_hash TEXT NOT NULL,
                dimension INTEGER NOT NULL, vector BLOB NOT NULL)""")

    def _connect(self):
        return sqlite3.connect(self.db_path)

    @staticmethod
    def _eligible(records: list[dict], folder: str | None) -> list[dict]:
        return [record for record in records if in_folder(record, folder)]

    def _check_model(self, db, model_id: str):
        row = db.execute("SELECT value FROM metadata WHERE key='model_id'").fetchone()
        if row and row[0] != model_id:
            raise ValueError(f"Index uses {row[0]}; requested {model_id}. Use a separate database.")
        db.execute("INSERT OR REPLACE INTO metadata VALUES ('model_id', ?)", (model_id,))

    def build(self, records: list[dict], backend, folder: str | None = None) -> dict:
        selected = self._eligible(records, folder)
        embedded = skipped = without_video = 0
        with self._connect() as db:
            self._check_model(db, backend.model_id)
            for record in selected:
                video = record.get("video")
                if not video or not Path(video).is_file():
                    without_video += 1
                    continue
                stat = Path(video).stat()
                description = f"{record.get('semantic_name', '')}: {record.get('description', '')}"
                digest = hashlib.sha256(description.encode()).hexdigest()
                old = db.execute("SELECT video,size,mtime_ns,description_hash FROM vectors WHERE asset_id=?", (record["asset_id"],)).fetchone()
                signature = (video, stat.st_size, stat.st_mtime_ns, digest)
                if old == signature:
                    skipped += 1
                    continue
                vector = _unit(backend.embed_video(video, record.get("semantic_name", ""), record.get("description", "")))
                db.execute("INSERT OR REPLACE INTO vectors VALUES (?,?,?,?,?,?,?,?)", (
                    record["asset_id"], record["category"], video, stat.st_size, stat.st_mtime_ns,
                    digest, vector.size, vector.tobytes(),
                ))
                db.commit()  # Keep progress if a later asset fails.
                embedded += 1
        return {"selected": len(selected), "embedded": embedded, "unchanged": skipped, "without_video": without_video}

    def search(self, query: str, records: list[dict], backend, top_k: int = 20, folder: str | None = None) -> dict:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        selected = self._eligible(records, folder)
        candidates = {record["asset_id"]: record for record in selected}
        without_video = sum(not record.get("video") or not Path(record["video"]).is_file() for record in selected)
        with self._connect() as db:
            self._check_model(db, backend.model_id)
            rows = []
            for row in db.execute("SELECT asset_id,video,size,mtime_ns,description_hash,dimension,vector FROM vectors"):
                record = candidates.get(row[0])
                if not record or not record.get("video") or not Path(record["video"]).is_file():
                    continue
                stat = Path(record["video"]).stat()
                description = f"{record.get('semantic_name', '')}: {record.get('description', '')}"
                signature = (record["video"], stat.st_size, stat.st_mtime_ns, hashlib.sha256(description.encode()).hexdigest())
                if row[1:5] == signature:
                    rows.append((row[0], row[5], row[6]))
        results = []
        if rows:
            query_vector = _unit(backend.embed_text(query))
            for asset_id, dimension, raw in rows:
                if dimension != query_vector.size:
                    raise ValueError(f"Vector dimension mismatch for {asset_id}")
                vector = np.frombuffer(raw, dtype=np.float32)
                results.append({**candidates[asset_id], "score": float(np.dot(query_vector, vector))})
            results.sort(key=lambda result: result["score"], reverse=True)
        indexed = len(rows)
        return {
            "query": query,
            "results": results[:top_k],
            "coverage": {"selected": len(selected), "indexed": indexed, "without_video": without_video,
                         "complete": indexed == len(selected) and without_video == 0},
        }
