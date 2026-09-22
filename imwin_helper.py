import threading
import queue
import time
import random
import numpy as np

from image_helper import Imgtool


class FPS_Booster:
    """
    抓圖快、處理慢的非同步管線。

    FPS 使用 EMA 平滑：
        fps = beta * fps + (1 - beta) * (1 / dt)

    beta 由使用者設定（0 < beta < 1）：
        beta → 1  越平滑、越遲鈍
        beta → 0  越靈敏、越抖動
    預設 0.9

    proc fps 採「固定時間窗取樣」：
        由 worker 完成事件累加計數，每 sample_interval 秒結算一次，
        與 consumer 呼叫 get() 的節奏解耦，任務長度不影響穩定度。
    """

    def __init__(self, workers=5, raw_maxsize=40, out_maxsize=40,
                 beta=0.9, sample_interval=0.5):
        self.func = None
        self._stop = False
        self.beta = float(beta)                    # ← 開放使用者設置
        self.sample_interval = float(sample_interval)

        self.raw_que = queue.Queue(maxsize=raw_maxsize)
        self.out_que = queue.Queue(maxsize=out_maxsize)

        # EMA 平滑 FPS
        now = time.time()
        self.put_fps = 0.0
        self.proc_fps = 0.0
        self.drop_fps = 0.0
        self._last_put_t = now
        self._last_proc_t = now
        self._last_drop_t = now
        self._eps = 1e-9

        # proc 事件計數（worker 端累加，get() 端取樣）
        self._proc_lock = threading.Lock()
        self._proc_count = 0

        # 累計
        self.total_put = 0
        self.total_proc = 0
        self.total_drop = 0

        # 啟動 workers（加入 startup jitter，避免 lockstep）
        self._workers = []
        for i in range(int(workers)):
            delay = i * 0.02 + random.uniform(0.0, 0.02)
            t = threading.Thread(
                target=self._worker_loop,
                args=(delay,),
                daemon=True,
            )
            t.start()
            self._workers.append(t)

    # ─────────────── 對外 API ───────────────
    def set_beta(self, beta):
        """設置 EMA 平滑係數，0 < beta < 1。"""
        self.beta = float(beta)
        return self

    def set_sample_interval(self, sec):
        """設置 proc fps 的取樣時間窗（秒）。"""
        self.sample_interval = float(sec)
        return self

    def set(self, func):
        self.func = func
        return self

    def put(self, img):
        now = time.time()
        self.put_fps, self._last_put_t = self._ema(
            self.put_fps, now, self._last_put_t)

        try:
            self.raw_que.put_nowait(img)
        except queue.Full:
            try:
                self.raw_que.get_nowait()
            except queue.Empty:
                pass
            self.drop_fps, self._last_drop_t = self._ema(
                self.drop_fps, now, self._last_drop_t)
            self.total_drop += 1
            try:
                self.raw_que.put_nowait(img)
            except queue.Full:
                pass
        self.total_put += 1

    def get(self):
        now = time.time()

        # ── proc fps：固定時間窗取樣，與 consumer 節奏無關 ──
        dt = now - self._last_proc_t
        if dt >= self.sample_interval:
            with self._proc_lock:
                n = self._proc_count
                self._proc_count = 0
            if n > 0:
                inst = n / dt
                if self.proc_fps <= 0.0:
                    self.proc_fps = inst
                else:
                    self.proc_fps = (self.beta * self.proc_fps
                                     + (1.0 - self.beta) * inst)
                self.total_proc += n
            self._last_proc_t = now

        # ── out_que 只負責取最新一張，不參與 fps 統計 ──
        result = None
        while True:
            try:
                result = self.out_que.get_nowait()
            except queue.Empty:
                break
        return result

    def stop(self):
        self._stop = True
        for t in self._workers:
            t.join(timeout=0.5)

    # ─────────────── 內部 ───────────────
    def _ema(self, old_fps, now, last_t):
        dt = now - last_t
        if dt < self._eps:
            return old_fps, last_t
        inst = 1.0 / dt
        if old_fps <= 0.0:
            return inst, now                     # 首次直接設值
        return (self.beta * old_fps
                + (1.0 - self.beta) * inst), now

    def _worker_loop(self, startup_delay=0.0):
        if startup_delay > 0:
            time.sleep(startup_delay)

        while not self._stop:
            try:
                img = self.raw_que.get(timeout=0.1)
            except queue.Empty:
                continue
            if self.func is None:
                continue
            try:
                out = self.func(img)
            except Exception as e:
                print(f"\n[FPS_Booster] worker error: {e}")
                continue

            # ★ 完成事件先記在 worker 端，不受 consumer 節奏影響
            with self._proc_lock:
                self._proc_count += 1

            try:
                self.out_que.put_nowait(out)
            except queue.Full:
                try:
                    self.out_que.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.out_que.put_nowait(out)
                except queue.Full:
                    pass


# ============================================================
# 模擬函式
# ============================================================
def simu_grab():
    for i in range(5):
        np.random.randint(0, 256, (680, 420, 3), dtype=np.uint8)
    return np.random.randint(0, 256, (680, 420, 3), dtype=np.uint8)


def simu_convert(img):
    im = Imgtool(img.copy())
    h, w = im.img.shape[:2]
    for _ in range(2000):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        r = np.random.randint(5, 50)
        im.circle((x, y, r), color='crimson', ld=-1)
    return im.get()


# ============================================================
# 主程式
# ============================================================
if __name__ == '__main__':
    WORKERS = 30
    BETA = 0.9
    SAMPLE_INTERVAL = 0.5          # proc fps 取樣窗（秒）

    booster = (FPS_Booster(workers=WORKERS,
                           beta=BETA,
                           sample_interval=SAMPLE_INTERVAL)
               .set(simu_convert))

    frame_id = 0
    _print_t = time.time()
    t_start = time.time()

    print(f"FPS_Booster 啟動：workers={WORKERS}, "
          f"beta={BETA}, sample_interval={SAMPLE_INTERVAL}s")

    try:
        while True:
            booster.put(simu_grab())
            frame_id += 1

            booster.get()          # 非阻塞

            now = time.time()
            if now - _print_t >= 1.0:
                _print_t = now
                rq = booster.raw_que.qsize()
                oq = booster.out_que.qsize()
                elapsed = now - t_start

                line = (
                    f"[{elapsed:6.1f}s] "
                    f"抓圖 fps={booster.put_fps:7.1f} | "
                    f"處理 fps={booster.proc_fps:6.1f} | "
                    f"丟棄 fps={booster.drop_fps:7.1f} | "
                    f"raw {rq}/{booster.raw_que.maxsize} "
                    f"out {oq}/{booster.out_que.maxsize} | "
                    f"累計 {booster.total_proc}"
                )
                print(f"\r{line:<100}", end='', flush=True)

    except KeyboardInterrupt:
        print("\n[stop] Ctrl+C")
    finally:
        booster.stop()
        print(f"\n總共抓 {booster.total_put} 張，"
              f"處理完 {booster.total_proc} 張，"
              f"丟棄 {booster.total_drop} 張")