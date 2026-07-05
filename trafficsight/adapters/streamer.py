"""Resilient HLS streamer backed by ``ffmpeg``.

The streamer spawns ``ffmpeg`` as a subprocess that decodes the HLS feed
into a stream of raw BGR24 frames. Frames are pushed onto a bounded
``queue.Queue``; if the consumer falls behind, the oldest queued frames
are silently dropped by ``Queue.put(timeout=0)`` to keep latency bounded.

If ``ffmpeg`` exits (network drop, server 503, ...), the reader thread
kills the process, sleeps 2 s, and respawns it. The consumer sees a
``queue.Empty`` exception during the gap and can retry.
"""

from __future__ import annotations

import queue
import subprocess
import threading
import time
from typing import Optional

from ..logger import write_log


class StableStreamer:
    """Thread-safe HLS → frame queue pump with auto-reconnect."""

    def __init__(self, src: str, width: int, height: int, fps: float,
                 buffer_seconds: int = 60) -> None:
        self.src = src
        self.width = width
        self.height = height
        self.fps = fps
        self.buffer_size = int(fps * buffer_seconds)
        self.q: queue.Queue = queue.Queue(maxsize=self.buffer_size)
        self.stopped = False
        self.proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

        self._cmd = [
            "ffmpeg",
            "-reconnect", "1",
            "-reconnect_streamed", "1",
            "-reconnect_delay_max", "10",
            "-i", src,
            "-vsync", "passthrough",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}",
            "-loglevel", "error",
            "pipe:1",
        ]

    # ----- lifecycle --------------------------------------------------------
    def start(self) -> "StableStreamer":
        self._spawn_process()
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self.stopped = True
        with self._lock:
            if self.proc is not None:
                self.proc.kill()
                self.proc = None

    # ----- consumer API -----------------------------------------------------
    def read(self, timeout: float = 10.0):
        """Block up to ``timeout`` seconds for the next frame."""
        return self.q.get(timeout=timeout)

    def queue_size(self) -> int:
        return self.q.qsize()

    # ----- internals --------------------------------------------------------
    def _spawn_process(self) -> None:
        self.proc = subprocess.Popen(
            self._cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10 ** 8,
        )

    def _pump(self) -> None:
        import numpy as np

        frame_size = self.width * self.height * 3
        while not self.stopped:
            proc = self.proc
            if proc is None or proc.stdout is None:
                time.sleep(0.5)
                continue

            raw = proc.stdout.read(frame_size)
            if len(raw) != frame_size:
                write_log("Koneksi stream putus, mencoba reconnect...")
                self._reconnect()
                continue

            frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                (self.height, self.width, 3)
            )
            try:
                self.q.put(frame, timeout=5.0)
            except queue.Full:
                # Consumer is too slow — drop this frame to keep latency low.
                pass

    def _reconnect(self) -> None:
        with self._lock:
            if self.proc is not None:
                self.proc.kill()
                self.proc = None
        time.sleep(2)
        if not self.stopped:
            self._spawn_process()
