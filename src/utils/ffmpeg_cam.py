"""FFmpeg fallback camera for Windows Store Python where cv2.VideoCapture is blocked."""
import subprocess, numpy as np, shutil, threading, time

class FFmpegCamera:
    def __init__(self, device="USB2.0 HD UVC WebCam", width=640, height=480, fps=30):
        self.w, self.h = width, height
        self.proc = None
        ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
        # dshow device
        cmd = [
            ffmpeg, "-f", "dshow",
            "-i", f'video={device}',
            "-vf", f"scale={width}:{height}",
            "-r", str(fps),
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "pipe:1"
        ]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=width*height*3*2)
        self.opened = self.proc is not None

    def isOpened(self):
        return self.opened and self.proc.poll() is None

    def read(self):
        if not self.isOpened():
            return False, None
        n = self.w*self.h*3
        raw = self.proc.stdout.read(n)
        if len(raw) < n:
            return False, None
        frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.h, self.w, 3))
        return True, frame.copy()

    def release(self):
        try:
            if self.proc:
                self.proc.terminate()
                self.proc.wait(timeout=3)
        except: pass
        try:
            if self.proc: self.proc.kill()
        except: pass

    def get(self, prop):
        import cv2
        if prop == cv2.CAP_PROP_FRAME_WIDTH: return self.w
        if prop == cv2.CAP_PROP_FRAME_HEIGHT: return self.h
        if prop == cv2.CAP_PROP_FPS: return 30
        return 0

    def set(self, *a, **k):
        return True
    def grab(self):
        return True


class ThreadedCamera:
    """Non-blocking wrapper: background thread keeps only latest frame. Fixes stuck."""
    def __init__(self, cap, max_q=1):
        self.cap = cap
        self.latest = None
        self.ret = False
        self.running = True
        self.lock = threading.Lock()
        self.stuck_count = 0
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def _loop(self):
        while self.running:
            try:
                r, f = self.cap.read()
            except Exception:
                r, f = False, None
            with self.lock:
                self.ret = r
                self.latest = f
                if not r:
                    self.stuck_count += 1
            # don't spin: match ~30fps capture
            time.sleep(0.005)
            if self.stuck_count > 200:
                time.sleep(0.5)
                self.stuck_count = 0

    def isOpened(self):
        try:
            return self.cap.isOpened()
        except: return True

    def read(self):
        with self.lock:
            if self.latest is None:
                return False, None
            return self.ret, self.latest.copy()

    def get(self, p):
        try: return self.cap.get(p)
        except: return 0

    def set(self, *a, **k):
        try: return self.cap.set(*a, **k)
        except: return True

    def grab(self):
        return True

    def release(self):
        self.running = False
        try: self.cap.release()
        except: pass
