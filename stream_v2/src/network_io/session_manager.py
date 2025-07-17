from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np
import time

@dataclass
class FaceSession:
    track_id: int
    start_frame: int
    frames: List[int] = field(default_factory=list)
    embeddings: List[np.ndarray] = field(default_factory=list)
    start_ts: float = field(default_factory=lambda: time.time())
    gender: str | None = None
    hijab: bool | None = None
    identity_id: int | None = None

    def add(self, frame_idx: int, emb: np.ndarray):
        self.frames.append(frame_idx)
        self.embeddings.append(emb)

class SessionManager:
    def __init__(self, maxlen=3):
        self._sessions: Dict[int, FaceSession] = {}
        self.maxlen = maxlen

    def update(self, track_id: int, frame_idx: int, emb: np.ndarray):
        if track_id not in self._sessions:
            self._sessions[track_id] = FaceSession(track_id=track_id, start_frame=frame_idx)
        sess = self._sessions[track_id]
        sess.add(frame_idx, emb)
        if len(sess.embeddings) > self.maxlen:
            sess.embeddings.pop(0)
            sess.frames.pop(0)

    def add_attributes(self, track_id: int, gender: str | None = None, hijab: bool | None = None):
        sess = self._sessions.get(track_id)
        if sess:
            if gender: sess.gender = gender
            if hijab:  sess.hijab = hijab

    def finalize(self, track_id: int) -> FaceSession | None:
        return self._sessions.pop(track_id, None)

    def get(self, track_id: int) -> FaceSession | None:
        return self._sessions.get(track_id)
