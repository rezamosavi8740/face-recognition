from src.config import CONFIG
from collections import OrderedDict
import asyncio 


class QueueDict:
    def __init__(self, maxlen: int):
        self.data = OrderedDict()
        self.maxlen = maxlen

    def add(self, key: int, value: float):
        if key in self.data:
            del self.data[key]
        self.data[key] = value
        if len(self.data) > self.maxlen:
            self.data.popitem(last=False)

    def get(self, key: int, default=0):
        return self.data.get(key, default)

    def __contains__(self, key: int):
        return key in self.data

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f"QueueDict({list(self.data.items())})"

class FaceCandidTrace:
    def __init__(self, maxlen: int = 150, logger=None, interval_frame_log=20):
        self.queue = QueueDict(maxlen)
        self.logger = logger
        self.last_log_frame = -1*interval_frame_log  # ensures first log happens immediately
        self.interval_frame_log = interval_frame_log
        self.fr_face_detection_th = CONFIG.fr.face_detection
        self.fr_face_surface_th = CONFIG.fr.face_surface

    def _should_log(self, current_frame):
        if (current_frame - self.last_log_frame) >= self.interval_frame_log:
            self.last_log_frame = current_frame
            return True
        return False

    def filter(self, result: dict):
        x1, y1, x2, y2 = result["face_bbox"]
        face_surface = (y2 - y1) * (x2 - x1)
        score = result["temp_face_score"]
        frame = result["frame_count"]

        if score < self.fr_face_detection_th  or face_surface < self.fr_face_surface_th:
            if self.logger and self._should_log(frame):
                self.logger.info(
                    f"Ignore instance — surface={face_surface}, score={score:.2f}, "
                    f"track_id={result['track_id']} @ {frame}"
                )
            return False
        return True

    def add(self, track_id: int, result: dict) -> bool:
        if self.filter(result):
            metric = result["temp_face_score"]
            prev = self.queue.get(track_id, 0)
            if (metric - 0.02) > prev:
                self.queue.add(track_id, metric)
                return True
            frame_count = result["frame_count"]
            if self.logger and self._should_log(frame_count):
                self.logger.info(
                    f"Ignore (better seen before): now={metric - 0.02:.2f}, "
                    f"history={prev:.2f}, track_id={result['track_id']} @ {frame_count}"
                )
        return False

    def get(self, track_id: int, default=0):
        return self.queue.get(track_id, default)

    def __contains__(self, track_id: int):
        return track_id in self.queue

    def __len__(self):
        return len(self.queue)

    def __repr__(self):
        return repr(self.queue)



class OverwriteQueue(asyncio.Queue):
    def __init__(self, maxsize, name, logger=None):
        super().__init__(maxsize=maxsize)
        self.logger = logger
        self.queue_name = name

    async def put(self, item):
        if self.full():
            _ = await self.get()
            if self.logger:
                self.logger.warning(f"[OverwriteQueue] Queue {self.queue_name} full — dropping oldest item.")
        await super().put(item)
