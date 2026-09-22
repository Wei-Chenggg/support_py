import csv
import pickle
import os
import shutil
import chardet
import time

def _fspath(p):
    # 將 PathLike / bytes 統一轉為 str，避免後續字串操作失敗
    if isinstance(p, bytes):
        return os.fsdecode(p)
    return os.fspath(p)
def _detect_encoding(fpath):
    # 使用 chardet 自動偵測檔案編碼，失敗時退回 utf-8
    with open(fpath, "rb") as f:
        raw = f.read()
    result = chardet.detect(raw)
    return result.get("encoding") or "utf-8"
def _check_exist_folder(fpath):
    # 檢查 fpath 的父資料夾是否存在，不存在則建立（可用於檔案或資料夾路徑）
    parent = os.path.dirname(os.path.abspath(fpath))
    if parent:
        os.makedirs(parent, exist_ok=True)
    return parent
def _ensure_ext(fpath, ext):
    # 若 fpath 結尾不是指定副檔名（不分大小寫），自動補上
    fpath = _fspath(fpath)
    if fpath.lower().endswith(ext.lower()):
        return fpath
    return fpath + ext
def _resolve_ext(fpath, ext):
    # 讀取用：優先使用原路徑，若不存在則嘗試補上副檔名
    fpath = _fspath(fpath)
    if os.path.exists(fpath):
        return fpath
    return _ensure_ext(fpath, ext)
def load_csv(fpath="./try.csv", encoding=None):
    # 讀取 CSV 檔案並回傳資料列表（自動補 .csv；encoding 為 None 時自動偵測）
    fpath = _resolve_ext(fpath, ".csv")
    if encoding is None:
        encoding = _detect_encoding(fpath)
    with open(fpath, "r", encoding=encoding, newline="") as f:
        return list(csv.reader(f))
def save_csv(fpath="./try.csv", data=None, encoding="utf-8-sig"):
    # 將資料寫入 CSV 檔案（自動補 .csv；參數順序顛倒會互換；父資料夾不存在會建立）
    if not isinstance(fpath, (str, bytes, os.PathLike)):
        fpath, data = data, fpath
    if data is None:
        data = []
    fpath = _ensure_ext(fpath, ".csv")
    _check_exist_folder(fpath)
    with open(fpath, "w", encoding=encoding, newline="") as f:
        csv.writer(f).writerows(data)
def append_csv(fpath="./try.csv", data=None, encoding="utf-8-sig"):
    # 將資料追加到 CSV 檔尾（檔案不存在則建立；自動補 .csv；父資料夾不存在會建立）
    if not isinstance(fpath, (str, bytes, os.PathLike)):
        fpath, data = data, fpath
    if data is None:
        data = []
    fpath = _ensure_ext(fpath, ".csv")
    _check_exist_folder(fpath)
    with open(fpath, "a", encoding=encoding, newline="") as f:
        csv.writer(f).writerows(data)
def load_pkl(fpath="./try.pkl"):
    # 讀取 pickle 檔案並回傳物件（自動補 .pkl）
    fpath = _resolve_ext(fpath, ".pkl")
    with open(fpath, "rb") as f:
        return pickle.load(f)
def save_pkl(fpath="./try.pkl", data=None):
    # 將物件以 pickle 格式寫入檔案（自動補 .pkl；參數順序顛倒會互換；父資料夾不存在會建立）
    if not isinstance(fpath, (str, bytes, os.PathLike)):
        fpath, data = data, fpath
    fpath = _ensure_ext(fpath, ".pkl")
    _check_exist_folder(fpath)
    with open(fpath, "wb") as f:
        pickle.dump(data, f)
def delete_file(fpath, ext=None, missing_ok=True):
    # 刪除檔案；ext 有給時自動補副檔名；missing_ok=True 則檔案不存在時不報錯
    if ext:
        fpath = _resolve_ext(fpath, ext)
    else:
        fpath = _fspath(fpath)
    try:
        os.remove(fpath)
        return True
    except FileNotFoundError:
        if missing_ok:
            return False
        raise
    except (IsADirectoryError, PermissionError):
        if os.path.isdir(fpath):
            raise IsADirectoryError(f"{fpath} 是資料夾，請改用 delete_folder")
        raise
def add_folder(fpath="./folder1"):
    # 建立資料夾（含多層），若已存在則不報錯
    if fpath:
        os.makedirs(fpath, exist_ok=True)
def _resolve_dst(src, dst):
    # 判斷 dst 是否為「資料夾意圖」，若是則建立資料夾並回傳 dst/原檔名
    # 判定為資料夾：結尾是路徑分隔符、已是現有資料夾、或整段路徑沒有副檔名
    dst = _fspath(dst)
    is_dir_intent = (
        dst.endswith(("/", "\\"))
        or os.path.isdir(dst)
        or not os.path.splitext(dst)[1]
    )
    if is_dir_intent:
        os.makedirs(dst, exist_ok=True)
        return os.path.join(dst, os.path.basename(_fspath(src)))
    return dst
def copy_file(src="./data.csv", dst=""):
    # 複製檔案（dst 為資料夾時會複製到該資料夾內，資料夾不存在會建立；dst 為空時不動作）
    if dst:
        dst = _resolve_dst(src, dst)
        _check_exist_folder(dst)
        shutil.copy2(src, dst)
def move_file(src="./data.csv", dst=""):
    # 移動檔案（dst 為資料夾時會移入該資料夾內，資料夾不存在會建立；dst 為空時不動作）
    if dst:
        dst = _resolve_dst(src, dst)
        _check_exist_folder(dst)
        shutil.move(src, dst)
def get_files(fpath="./folder1", types=["csv", "png", "txt"], accept=None, reject=None):
    # 取得資料夾內符合條件的檔案清單，回傳 [(index, path, name, sub_name), ...]
    # name 為主檔名（不含副檔名），sub_name 為副檔名（不含點）
    # types=None 或 [] 表示不限制副檔名；accept/reject 為檔名子字串篩選
    fpath = _fspath(fpath)
    file_info = []
    if not os.path.isdir(fpath):
        return file_info
    types_lower = [t.lower().lstrip(".") for t in types] if types else []
    for fn in sorted(os.listdir(fpath)):
        full = os.path.join(fpath, fn)
        if not os.path.isfile(full):
            continue
        if accept and accept not in fn:
            continue
        if reject and reject in fn:
            continue
        stem, ext = os.path.splitext(fn)
        sub_name = ext.lstrip(".")
        if types_lower and sub_name.lower() not in types_lower:
            continue
        file_info.append((len(file_info), full, stem, sub_name))
    return file_info
def copy_files(src, dst, types=["csv"], accept=None, reject=None):
    # 批次複製 src 資料夾內符合條件的檔案到 dst 資料夾（dst 不存在會建立）
    # 回傳複製成功的檔案清單 [(index, path, name, sub_name), ...]
    dst = _fspath(dst)
    os.makedirs(dst, exist_ok=True)
    files = get_files(src, types=types, accept=accept, reject=reject)
    copied = []
    for idx, path, name, sub_name in files:
        copy_file(path, os.path.join(dst, os.path.basename(path)))
        copied.append((len(copied), path, name, sub_name))
    return copied
def move_files(src, dst, types=["csv"], accept=None, reject=None):
    # 批次移動 src 資料夾內符合條件的檔案到 dst 資料夾（dst 不存在會建立）
    # 回傳移動成功的檔案清單 [(index, path, name, sub_name), ...]
    dst = _fspath(dst)
    os.makedirs(dst, exist_ok=True)
    files = get_files(src, types=types, accept=accept, reject=reject)
    moved = []
    for idx, path, name, sub_name in files:
        move_file(path, os.path.join(dst, os.path.basename(path)))
        moved.append((len(moved), path, name, sub_name))
    return moved
def get_timestamp():
    # 回傳當前時間字串（格式：YYYYMMDD-HHMMSS）
    return time.strftime('%Y%m%d-%H%M%S', time.localtime())