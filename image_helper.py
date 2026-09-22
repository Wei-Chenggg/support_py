import os
import functools
import numpy as np
import cv2
import matplotlib.colors as mcolors
from PIL import Image, ImageDraw, ImageFont


class Imgtool:
    """
    影像處理工具箱
    ── 內部格式公約 ─────────────────────────────────────
    * 通道順序一律 BGR（3 通道）或 BGRA（4 通道），與 OpenCV 一致
      → 與 PIL / matplotlib 互動時，只透過 _bgr_to_pil / _pil_to_bgr 轉換
    * dtype 一律 uint8
    * 顏色公開介面：字串名稱 / hex（走 matplotlib）、BGR(A) tuple、灰階純量
    * 中文路徑：所有檔案 I/O 走 open()，不用 cv2.imread / np.fromfile / np.tofile
    """
    # 字型候選（依序嘗試，第一個能載入的勝出）
    _FONT_CANDIDATES = (
        'msjh.ttc',                  # Windows 微軟正黑
        'msyh.ttc',                  # Windows 微軟雅黑
        'simhei.ttf',                # Windows 黑體
        'PingFang.ttc',              # macOS 蘋方
        'STHeiti Light.ttc',         # macOS 華文黑體
        'NotoSansCJK-Regular.ttc',   # Linux Noto CJK
        'WenQuanYiZenHei.ttf',       # Linux 文泉驛
    )

    def __init__(self, img, copy=False):
        self.load(img, copy)

    # ================================== [載入 / 輸出] ==================================
    def load(self, img, copy=False):
        """
        img 可為：
          - str / os.PathLike : 檔案路徑（支援中文）
          - np.ndarray        : 既有影像（預設不複製，copy=True 才複製）
          - tuple             : (w, h) 或 (w, h, c)，c ∈ {1, 3, 4}
        """
        if isinstance(img, (str, os.PathLike)):
            self.img = self._load_img(str(img))
        elif isinstance(img, np.ndarray):
            self.img = img.copy() if copy else img
        elif isinstance(img, tuple):
            self.img = self._create_img(img)
        else:
            raise TypeError(f"Unsupported img type: {type(img).__name__}")
        # 四角文字累積偏移
        self._acc = {k: 12 for k in
                     ('top-left', 'top-right', 'bottom-left', 'bottom-right')}
        return self

    def get(self):
        return self.img

    def save(self, path='./img.png'):
        """支援中文路徑；依副檔名推斷格式（預設 .png）"""
        ext = os.path.splitext(str(path))[1] or '.png'
        ok, buf = cv2.imencode(ext, self.img)
        if not ok:
            raise IOError(f"Failed to encode image: {path}")
        # 用 open() 寫，避開 np.tofile 的中文路徑問題
        with open(path, 'wb') as f:
            f.write(buf.tobytes())
        return self

    # ================================== [幾何：補白] ==================================
    def pad_by_pixel(self, t=100, b=100, l=100, r=100, color='black'):
        """四邊各補固定像素；color 可為名稱 / BGR(A) / 灰階純量"""
        h, w = self.img.shape[:2]
        c = self._fit_color(color)
        if self.img.ndim == 2:
            canvas = np.full((h + t + b, w + l + r), c, dtype=np.uint8)
            canvas[t:t + h, l:l + w] = self.img
        else:
            ch = self.img.shape[2]
            canvas = np.full((h + t + b, w + l + r, ch), c, dtype=np.uint8)
            canvas[t:t + h, l:l + w] = self.img
        self.img = canvas
        return self

    def pad_by_ratio(self, t=0.1, b=0.1, l=0.1, r=0.1, color='black'):
        h, w = self.img.shape[:2]
        return self.pad_by_pixel(int(h * t), int(h * b),
                                 int(w * l), int(w * r), color)

    # ================================== [影像處理] ==================================
    def smooth(self, size=5, sigma=0):
        """高斯模糊。size 自動修正為奇數；sigma=0 讓 OpenCV 依 size 自動推算。"""
        k = int(size)
        if k % 2 == 0:
            k += 1
        if k < 1:
            return self
        self.img = cv2.GaussianBlur(self.img, (k, k), float(sigma))
        return self

    def binary(self, val=127, otsu=False, invert=False):
        """
        二值化。內部先轉灰階，輸出單通道 (H, W) uint8，值為 {0, 255}。
        val    : 閾值（otsu=True 時忽略）
        otsu   : 是否用 Otsu 自動找閾值（需雙峰分佈才有效）
        invert : 是否反轉（前景黑、背景白）
        """
        gray = self._cvt_to_gray(self.img)
        flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        if otsu:
            _, out = cv2.threshold(gray, 0, 255, flag + cv2.THRESH_OTSU)
        else:
            _, out = cv2.threshold(gray, int(val), 255, flag)
        self.img = out
        return self

    def canny(self, low=50, high=150):
        """
        Canny 邊緣偵測。回傳 (H, W) uint8，值為 {0, 255}。不改動 self.img。
        low / high : 雙閾值，慣例 high ≈ 2~3 × low
        注意：本方法不做前置模糊，雜點圖請先自行 smooth() 再 canny()。
        """
        gray = self._cvt_to_gray(self.img)
        return cv2.Canny(gray, int(low), int(high))

    # ================================== [繪圖] ==================================
    def circle(self, circles, color='crimson', ld=1):
        """circles: (x, y, r) 或 [(x,y,r), ...]；ld=-1 為實心"""
        if circles is None:
            return self
        c = self._fit_color(color)
        for x, y, r in np.atleast_2d(circles):
            cv2.circle(self.img, (int(x), int(y)), int(r), c, int(ld))
        return self

    def line(self, pts, color='crimson', ld=1):
        """pts: (x1, y1, x2, y2) 或 [(x1,y1,x2,y2), ...]"""
        if pts is None:
            return self
        c = self._fit_color(color)
        for x1, y1, x2, y2 in np.atleast_2d(pts):
            cv2.line(self.img, (int(x1), int(y1)), (int(x2), int(y2)), c, int(ld))
        return self

    def rect(self, pts, color='crimson', ld=1):
        """pts: (x1, y1, x2, y2) 或 [(x1,y1,x2,y2), ...]；ld=-1 為實心"""
        if pts is None:
            return self
        c = self._fit_color(color)
        for x1, y1, x2, y2 in np.atleast_2d(pts):
            cv2.rectangle(self.img, (int(x1), int(y1)), (int(x2), int(y2)), c, int(ld))
        return self

    def poly(self, pts, color='crimson', ld=1, fill=False):
        """
        pts 支援三種輸入：
          單多邊形   : [[x,y], [x,y], ...]           → shape (N, 2)
          多邊形集合 : [[[x,y],...], [[x,y],...]]    → shape (M, N, 2)
          已 reshape : [[[x,y]], [[x,y]], ...]       → shape (N, 1, 2)
        fill=True 為填滿，False 為 closed 折線
        """
        if pts is None:
            return self
        c = self._fit_color(color)
        arr = np.asarray(pts)
        if arr.ndim == 2:
            contours = [arr.reshape(-1, 1, 2).astype(np.int32)]
        elif arr.ndim == 3 and arr.shape[1] == 1:
            contours = [arr.astype(np.int32)]
        elif arr.ndim == 3:
            contours = [a.reshape(-1, 1, 2).astype(np.int32) for a in arr]
        else:
            raise ValueError(f"Unsupported pts shape: {arr.shape}")
        if fill:
            cv2.fillPoly(self.img, contours, c)
        else:
            cv2.polylines(self.img, contours, isClosed=True,
                          color=c, thickness=int(ld))
        return self

    # ================================== [文字] ==================================
    @staticmethod
    @functools.lru_cache(maxsize=64)
    def _load_font(size, path=None):
        """
        載入字型：優先用 path（若給），否則依候選清單挑第一個能載入的。
        快取鍵為 (size, path)。
        """
        if path:
            try:
                return ImageFont.truetype(path, size)
            except (OSError, IOError):
                pass
        for name in Imgtool._FONT_CANDIDATES:
            try:
                return ImageFont.truetype(name, size)
            except (OSError, IOError):
                continue
        return ImageFont.load_default()

    def text(self, xy, text, color='crimson', fontsize=5,
             anchor='lt', font_path=None, ld=0):
        """
        在 xy 處繪製單行文字（支援中文）。
        xy     : 錨點位置 (x, y)
        anchor : 兩字元，水平 l/m/r + 垂直 t/m/b，例如：
                 lt(左上, 預設) mt rt
                 lm     mm rm
                 lb     mb rb
        fontsize : 字級，為影像高度的百分比
        ld     : 描邊寬度（給 PIL stroke_width，負值自動歸 0）
        """
        if not text:
            return self
        text = str(text)
        ld = max(0, int(ld))                    # ← 保護負值
        h_img, w_img = self.img.shape[:2]
        size = max(8, int(h_img * fontsize / 100))
        font = self._load_font(size, font_path)
        bbox = font.getbbox(text)              # (x0, y0, x1, y1)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        # 依 anchor 換算成左上角座標
        x, y = int(xy[0]), int(xy[1])
        a_h, a_v = anchor[0].lower(), anchor[1].lower()
        if a_h == 'm':   x -= tw // 2
        elif a_h == 'r': x -= tw
        if a_v == 'm':   y -= th // 2
        elif a_v == 'b': y -= th
        # 裁切到影像可見範圍
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w_img, x + tw), min(h_img, y + th)
        if x2 <= x1 or y2 <= y1:
            return self
        # 取出 ROI → 轉 PIL → 繪製 → 寫回
        roi = self.img[y1:y2, x1:x2]
        pil_roi = self._bgr_to_pil(roi)
        pil_color = self._pil_text_color(color)
        ImageDraw.Draw(pil_roi).text(
            (x - x1, y - y1), text,
            font=font,
            fill=pil_color,
            anchor='lt',
            stroke_width=ld,
            stroke_fill=pil_color if ld else None,
        )
        self.img[y1:y2, x1:x2] = self._pil_to_bgr(pil_roi)
        return self

    def auto_text(self, text, corner='top-left', color='black',
                  fontsize=5, ld=0, line_spacing=1.2, outside=False):
        """
        自動在四角堆疊文字。連續呼叫會依 _acc 往下（上）累加，不會重疊。
        corner  : 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right'
        outside : True 時先擴張畫布，把文字放在畫布外的留白區
        """
        if not text:
            return self
        ld = max(0, int(ld))                    # ← 保護負值
        key = corner.lower().replace('upper', 'top').replace('lower', 'bottom')
        if key not in self._acc:
            key = 'top-left'
        lines = text.split('\n') if isinstance(text, str) else [str(t) for t in text]
        H, W = self.img.shape[:2]
        pad = 12
        size = max(8, int(H * fontsize / 100))
        font = self._load_font(size)
        try:
            ascent, descent = font.getmetrics()
            line_h = int((ascent + descent) * line_spacing)
        except AttributeError:
            line_h = int(size * 1.4 * line_spacing)
        total_h = line_h * len(lines)
        # ── 決定 y_start ──────────────────────────
        if outside:
            if 'top' in key:
                self.pad_by_pixel(t=total_h + pad, b=0, l=0, r=0, color='white')
            else:
                self.pad_by_pixel(t=0, b=total_h + pad, l=0, r=0, color='white')
            H, W = self.img.shape[:2]
            y_start = pad if 'top' in key else H - pad - total_h
        else:
            if 'top' in key:
                y_start = self._acc[key]
            else:
                y_start = H - self._acc[key] - total_h
            self._acc[key] += total_h
        # ── ROI 裁切（只處理可見範圍）──────────────
        y1_roi = max(0, y_start)
        y2_roi = min(H, y_start + total_h)
        if y1_roi >= y2_roi:
            return self
        roi = self.img[y1_roi:y2_roi]
        pil_roi = self._bgr_to_pil(roi)
        draw = ImageDraw.Draw(pil_roi)
        pil_color = self._pil_text_color(color)
        for i, line in enumerate(lines):
            line = str(line)
            w_txt = int(draw.textlength(line, font=font))
            x = pad if 'left' in key else W - pad - w_txt
            y_roi = y_start + i * line_h - y1_roi
            draw.text(
                (x, y_roi), line,
                font=font, fill=pil_color, anchor='lt',
                stroke_width=ld,
                stroke_fill=pil_color if ld else None,
            )
        self.img[y1_roi:y2_roi] = self._pil_to_bgr(pil_roi)
        return self

    # ================================== [私有工具：色彩] ==================================
    def _c2bgr(self, c):
        """名稱/hex → BGR tuple；其他（BGR(A) tuple / 灰階純量）原樣回傳"""
        if isinstance(c, str):
            r, g, b = (int(x * 255) for x in mcolors.to_rgba(c)[:3])
            return (b, g, r)
        return c

    def _c2rgb(self, c):
        """名稱/hex → RGB tuple；灰階純量 → (v,v,v)；其他原樣回傳"""
        if isinstance(c, str):
            return tuple(int(x * 255) for x in mcolors.to_rgba(c)[:3])
        if isinstance(c, (int, float, np.integer, np.floating)):
            v = int(c)
            return (v, v, v)
        return tuple(c)

    def _fit_color(self, c):
        """
        依影像通道數調整顏色：
          灰階 (H,W)     → 標量（OpenCV 灰階權重）
          彩色 (H,W,3)   → BGR tuple
          彩色 (H,W,4)   → BGRA tuple（自動補 alpha=255）
        """
        c = self._c2bgr(c)
        # 灰階圖：tuple → 標量
        if self.img.ndim == 2:
            if isinstance(c, (tuple, list)):
                b, g, r = (list(c) + [0, 0, 0])[:3]
                return int(0.114 * b + 0.587 * g + 0.299 * r)
            return int(c)
        # BGRA 圖：3 元素補 alpha=255，避免 OpenCV 補 0 變透明
        if self.img.shape[2] == 4:
            if isinstance(c, (tuple, list)):
                return tuple((list(c) + [0, 0, 0])[:3]) + (255,)
            v = int(c)
            return (v, v, v, 255)
        return c

    def _pil_text_color(self, color):
        """PIL 填色：彩色影像 → RGB(A) tuple；灰階影像 → int"""
        rgb = self._c2rgb(color)
        if self.img.ndim == 2:
            r, g, b = rgb
            return int(0.299 * r + 0.587 * g + 0.114 * b)
        if self.img.shape[2] == 4:
            return tuple(rgb) + (255,)
        return rgb

    # ================================== [私有工具：通道轉換] ==================================
    @staticmethod
    def _cvt_to_gray(img):
        """
        BGR / BGRA / 灰度 ndarray → 單通道灰度 (H, W) uint8
        - 已是灰度 (ndim==2)：原樣返回（不複製）
        - BGRA (4ch)：BGRA2GRAY
        - BGR  (3ch)：BGR2GRAY
        """
        if img.ndim == 2:
            return img
        ch = img.shape[2]
        if ch == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        if ch == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        raise ValueError(f"_cvt_to_gray: unsupported channels = {ch}")

    @staticmethod
    def _cvt_to_color(img):
        """
        灰度 / BGR / BGRA ndarray → BGR 或 BGRA (H, W, 3 or 4) uint8
        - 灰度 (ndim==2)：GRAY2BGR，複製成 3 通道
        - BGR / BGRA：原樣返回（不複製）
        """
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        ch = img.shape[2]
        if ch in (3, 4):
            return img
        raise ValueError(f"_cvt_to_color: unsupported channels = {ch}")

    # ================================== [私有工具：BGR ↔ PIL] ==================================
    @staticmethod
    def _bgr_to_pil(roi):
        """BGR / BGRA / 灰階 ndarray → PIL Image（RGB / RGBA / L）"""
        if roi.ndim == 2:
            return Image.fromarray(roi)
        if roi.shape[2] == 4:
            return Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGRA2RGBA))
        return Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

    @staticmethod
    def _pil_to_bgr(pil_img):
        """PIL Image → BGR / BGRA / 灰階 ndarray"""
        arr = np.array(pil_img)
        if arr.ndim == 2:
            return arr
        if arr.shape[2] == 4:
            return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    # ================================== [私有工具：載入] ==================================
    @staticmethod
    def _load_img(path):
        """用 open() 讀 bytes（支援中文路徑），再交給 cv2.imdecode 解碼"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Img not found: {path}")
        with open(path, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to decode image: {path}")
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        return img

    @staticmethod
    def _create_img(spec, color=255):
        """spec: (w, h) 或 (w, h, c)，c ∈ {1, 3, 4}"""
        if len(spec) == 2:
            w, h, c = spec[0], spec[1], 3
        elif len(spec) == 3:
            w, h, c = spec
        else:
            raise ValueError(f"Invalid shape spec: {spec}")
        if c == 1:
            return np.full((h, w), color, dtype=np.uint8)
        return np.full((h, w, c), color, dtype=np.uint8)


# ======================================== [測試] ========================================
if __name__ == '__main__':
    OUT_DIR = './_imgtool_demo'
    os.makedirs(OUT_DIR, exist_ok=True)

    im = Imgtool("./123.png")
    print("shape:", im.img.shape, "dtype:", im.img.dtype)
    print("ndim:", im.img.ndim)
    print("crimson as BGR:", im._c2bgr('crimson'))
    print("fit for this img:", im._fit_color('crimson'))
    im.auto_text('左上 corner', corner='top-left', color='red', fontsize=3)
    im.rect((20, 20, 300, 200), color='yellow', ld=6)
    im.save("./try.png")

    # ── 1. 彩色：圖形 + 中文 + 四角 ─────────────────────────
    im = Imgtool((800, 600))
    im.img[:] = 255
    im.circle((100, 100, 50), color='crimson', ld=2)
    im.circle([(250, 100, 40), (350, 100, 40)], color='steelblue', ld=-1)
    im.line((50, 200, 400, 200), color='green', ld=3)
    im.rect((50, 250, 200, 350), color='orange', ld=2)
    im.rect((250, 250, 400, 350), color='purple', ld=-1)
    im.poly([(50, 400), (150, 400), (100, 480)], color='teal', fill=True)
    im.poly([(200, 400), (300, 420), (280, 480), (210, 470)], color='navy', ld=2)
    im.poly([
        [(450, 400), (550, 400), (500, 480)],
        [(450, 400), (550, 400), (500, 320)],
    ], color='coral', fill=True)
    im.text((50, 520), '中文測試 Text', color='black', fontsize=8)
    im.text((50, 560), '描邊文字', color='white', fontsize=8, ld=2)
    im.auto_text('左上 corner', corner='top-left', color='red', fontsize=3)
    im.auto_text('右上 corner', corner='top-right', color='blue', fontsize=3)
    im.auto_text('左下 corner', corner='bottom-left', color='green', fontsize=3)
    im.auto_text('右下 corner 1', corner='bottom-right', color='purple', fontsize=3)
    im.auto_text('右下 corner 2', corner='bottom-right', color='purple', fontsize=3)
    im.auto_text('右下 corner 3', corner='bottom-right', color='purple', fontsize=3)
    p1 = os.path.join(OUT_DIR, '測試輸出_中文.png')
    im.save(p1)
    print(f'saved → {p1}')

    # ── 2. 灰階 ───────────────────────────────────────────
    im_g = Imgtool((400, 300, 1))
    im_g.img[:] = 255
    im_g.circle((100, 100, 50), color='crimson', ld=-1)
    im_g.line((20, 200, 380, 200), color=(0, 0, 0), ld=2)
    im_g.text((20, 250), '灰階 Gray', color='black', fontsize=8)
    p2 = os.path.join(OUT_DIR, '測試輸出_灰階.png')
    im_g.save(p2)
    print(f'saved → {p2}')

    # ── 3. outside 擴畫布 ────────────────────────────────
    im3 = Imgtool((400, 300))
    im3.img[:] = 230
    im3.rect((100, 100, 300, 200), color='gray', ld=-1)
    im3.auto_text('外推標題 Title', corner='top-left',
                  color='black', fontsize=5, outside=True)
    im3.auto_text('外推說明 subtitle', corner='top-left',
                  color='gray', fontsize=4, outside=True)
    p3 = os.path.join(OUT_DIR, '測試輸出_outside.png')
    im3.save(p3)
    print(f'saved → {p3}  shape={im3.img.shape}')

    # ── 4. smooth / binary / canny ────────────────────────
    base = Imgtool((400, 400))
    base.img[:] = 255
    base.circle((120, 120, 60), color='black', ld=-1)
    base.rect((230, 60, 350, 180), color='black', ld=-1)
    rng = np.random.default_rng(0)
    noise_mask = rng.random(base.img.shape[:2]) < 0.03
    base.img[noise_mask] = 0
    base.save(os.path.join(OUT_DIR, '00_原始含雜點.png'))

    # 先 smooth 再 canny：乾淨邊緣
    smoothed = Imgtool(base.img, copy=True).smooth(size=5)
    edges_low = smoothed.canny(low=30, high=90)
    edges_hi = smoothed.canny(low=100, high=200)
    cv2.imwrite(os.path.join(OUT_DIR, '05_canny_30_90_smooth.png'), edges_low)
    cv2.imwrite(os.path.join(OUT_DIR, '06_canny_100_200_smooth.png'), edges_hi)

    # 直接 canny：雜點會爆
    edges_raw = Imgtool(base.img, copy=True).canny(low=50, high=150)
    cv2.imwrite(os.path.join(OUT_DIR, '07_canny_noblur.png'), edges_raw)

    # 對照組：smooth / binary
    Imgtool(base.img, copy=True).smooth(size=7).save(
        os.path.join(OUT_DIR, '01_smooth_7x7.png'))
    Imgtool(base.img, copy=True).binary(val=127).save(
        os.path.join(OUT_DIR, '02_binary_val127.png'))
    Imgtool(base.img, copy=True).binary(otsu=True).save(
        os.path.join(OUT_DIR, '03_binary_otsu.png'))
    Imgtool(base.img, copy=True).binary(val=127, invert=True).save(
        os.path.join(OUT_DIR, '04_binary_invert.png'))

    print(f'canny 回傳 type={type(edges_low).__name__}, '
          f'shape={edges_low.shape}, dtype={edges_low.dtype}, '
          f'unique={np.unique(edges_low)}')
    ys, xs = np.where(edges_low > 0)
    pts = np.column_stack([xs, ys])
    print(f'邊緣點數 N={len(pts)}, pts.shape={pts.shape}, 前 3 點={pts[:3].tolist()}')
    contours, _ = cv2.findContours(edges_low.copy(),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print(f'findContours 抓到 {len(contours)} 條輪廓，'
          f'各輪廓點數={[len(c) for c in contours]}')

    # ── 5. 中文路徑 round-trip ────────────────────────────
    p5 = os.path.join(OUT_DIR, '中文檔名_讀取測試.png')
    Imgtool(p1).save(p5)
    reloaded = Imgtool(p5)
    print(f'reloaded ← {p5}  shape={reloaded.img.shape}')
    print(f'\n全部完成，請打開 {OUT_DIR}')