import subprocess
import numpy as np
import queue
import threading
import time
from logger import write_log


class StableStreamer:
    def __init__(self, src, width, height, fps):
        self.src = src
        self.width = width
        self.height = height
        self.fps = fps
        self.buffer_size = int(fps * 60)
        self.q = queue.Queue(maxsize=self.buffer_size)
        self.stopped = False
        self.proc = None
        self._lock = threading.Lock()
        self.cmd = [
            'ffmpeg', '-reconnect', '1', '-reconnect_streamed', '1',
            '-reconnect_delay_max', '10', '-i', src,
            '-vsync', 'passthrough', '-f', 'rawvideo',
            '-pix_fmt', 'bgr24', '-s', f'{width}x{height}',
            '-loglevel', 'error', 'pipe:1'
        ]

    def start(self):
        self.proc = subprocess.Popen(
            self.cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10**8,
        )
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        return self

    def _update(self):
        frame_size = self.width * self.height * 3
        while not self.stopped:
            raw = self.proc.stdout.read(frame_size)
            if len(raw) != frame_size:
                write_log("Koneksi putus, mencoba reconnect...")
                with self._lock:
                    if self.proc:
                        self.proc.kill()
                time.sleep(2)
                with self._lock:
                    self.proc = subprocess.Popen(
                        self.cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        bufsize=10**8,
                    )
                continue

            frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3))
            try:
                self.q.put(frame, timeout=5.0)
            except queue.Full:
                pass

    def read(self, timeout=10.0):
        return self.q.get(timeout=timeout)

    def queue_size(self):
        return self.q.qsize()

    def stop(self):
        self.stopped = True
        with self._lock:
            if self.proc:
                self.proc.kill()
