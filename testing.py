import os
import json
import time
from pathlib import Path
import argparse
import glob
import pickle

import numpy as np
import torch
import librosa
import matplotlib.pyplot as plt  # 若你不需要顯示圖，其實不必 import
import soundfile as sf

import model  # 你的 CRNN2D_elu2 定義

# --------- 參數（與訓練一致）---------
SR = 16000
N_MELS = 128
N_FFT = 2048
HOP = 512
SLICE_LEN = 157           # 每片時間框數（和你訓練/驗證一致）
RNN_HIDDEN = 32           # 你的 GRU 隱層大小（和訓練一致）
BATCH_CHUNK = 128         # 一次丟給 GPU 的切片數，避免 OOM，可調

# --------- 路徑 ---------
BEST_DIR = Path('/content/drive/MyDrive/CRNN_runs/best')
BEST_MODEL_DIR = BEST_DIR / 'model'
BEST_LABEL_PATH = BEST_DIR / 'label_encoder.pkl'   # 如果有存就會用
TEST_DIR = Path('/content/drive/MyDrive/artist20/test')
OUT_JSON = BEST_DIR / 'result' / 'test_top3.json'  # 產出位置


# test_ROOT = Path(__file__).resolve().parent
# BEST_DIR = test_ROOT / 'best3'
# ROOT = test_ROOT.parent
# TEST_DIR = ROOT / 'artist20' / 'test'

def find_best_ckpt(model_dir: Path) -> Path:
    # 優先找 *.pt；找不到就取資料夾下第一個檔案
    pt_list = sorted(model_dir.glob('*.pt'))
    if len(pt_list) > 0:
        return pt_list[0]
    files = [p for p in model_dir.iterdir() if p.is_file()]
    if not files:
        raise FileNotFoundError(f'No checkpoint found under: {model_dir}')
    return files[0]

def build_model(classes_num: int, gid: int = 0):
    device = torch.device(f'cuda:{gid}' if torch.cuda.is_available() else 'cpu')
    net = model.CRNN2D_elu2(288, classes_num)  # 與你訓練一致
    net.float().to(device)
    net.eval()
    return net, device

# --- 讀取 ckpt（用 CPU 讀即可），吐回 ckpt 物件與 classes_num ---
def load_checkpoint(ckpt_path, device):
    ckpt = torch.load(str(ckpt_path), map_location=device)   # 保持原本做法即可
    classes_num = int(ckpt.get('classes_num', 20))
    model_name  = ckpt.get('model_name', 'CRNN2D_elu')
    state_dict  = ckpt['state_dict']
    return ckpt, state_dict, classes_num, model_name

# --- 依 ckpt 的 model_name 建立正確的模型類別 ---
def build_model_from_ckpt(model_name: str, classes_num: int, gid: int = 0):
    device = torch.device(f'cuda:{gid}' if torch.cuda.is_available() else 'cpu')

    # 依你專案內的可用類別分派
    if model_name == 'CRNN2D_elu':
        net = model.CRNN2D_elu(288, classes_num)
    elif model_name == 'CRNN2D_elu2':
        net = model.CRNN2D_elu2(288, classes_num)
    else:
        raise ValueError(f"Unknown model_name in ckpt: {model_name}")

    net.float().to(device)
    net.eval()
    return net, device

def load_label_names(label_pkl_path: Path, classes_num: int):
    if label_pkl_path.exists():
        try:
            with open(label_pkl_path, 'rb') as f:
                le = pickle.load(f)
            class_names = list(le.classes_)
            if len(class_names) != classes_num:
                print(f'[warn] label_encoder classes ({len(class_names)}) != classes_num ({classes_num}); fallback to index names.')
                class_names = [f'class_{i}' for i in range(classes_num)]
            return class_names
        except Exception as e:
            print('[warn] 讀取 label_encoder 失敗：', e)
    # fallback
    return [f'class_{i}' for i in range(classes_num)]

def song_to_slices(song_path: Path):
    # 做成和你訓練一致的 log-mel，切成固定寬度 SLICE_LEN 的片段
    y, sr = librosa.load(str(song_path), sr=SR, mono=True) 
    S = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP) 
    log_S = librosa.core.amplitude_to_db(S, ref=1.0) 

    T = log_S.shape[1]
    slices = T // SLICE_LEN
    X_list = []
    if slices == 0:
        # 太短：右邊 padding 到 SLICE_LEN
        pad = SLICE_LEN - T
        pad_block = np.pad(log_S, ((0,0),(0,pad)), mode='edge')
        X_list.append(pad_block[:, :SLICE_LEN])
    else:
        for j in range(slices):
            X_list.append(log_S[:, SLICE_LEN*j : SLICE_LEN*(j+1)])

    X = np.stack(X_list, axis=0)  # [num_slices, N_MELS, SLICE_LEN]
    return X

@torch.no_grad()
def infer_one_song(net, device, x_np: np.ndarray):
    # 可能切片很多，分批送進去避免 OOM
    num = x_np.shape[0]
    probs_sum = None
    for s in range(0, num, BATCH_CHUNK):
        e = min(s + BATCH_CHUNK, num)
        xb = torch.from_numpy(x_np[s:e]).float().to(device)         # [B, 128, 157]
        hb = torch.randn(1, xb.size(0), RNN_HIDDEN, device=device)  # [1, B, 32]

        logits, _ = net(xb, hb)            # [B, C]
        probs = torch.softmax(logits, dim=1)  # 分類用 softmax 機率
        probs_sum = probs if probs_sum is None else probs_sum + probs

    probs_mean = probs_sum / num
    probs_mean = probs_mean.mean(dim=0)  # 跨切片平均後再對歌曲平均（也可直接 sum 後 argmax，差異不大）
    return probs_mean.cpu().numpy()      # [C]

def main(gid: int = 0):
    t0 = time.time()

    # 1) 找 checkpoint、載入
    ckpt_path = find_best_ckpt(BEST_MODEL_DIR)
    print('[info] 使用 checkpoint:', ckpt_path)

    # 先在 CPU 讀 ckpt
    ckpt, state_dict, classes_num, model_name = load_checkpoint(ckpt_path, device=torch.device('cpu'))
    print('[info] ckpt.model_name =', model_name, ' | classes_num =', classes_num)

    # 建立對的網路類別
    net, device = build_model_from_ckpt(model_name=model_name, classes_num=classes_num, gid=gid)
    net.load_state_dict(state_dict)
    class_names = load_label_names(BEST_LABEL_PATH, classes_num)
    print('[info] 類別數:', classes_num)

    # 2) 讀 test 檔名（僅音檔）
    exts = ('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aiff', '.aif', '.au')
    files = sorted([p for p in TEST_DIR.iterdir() if p.suffix.lower() in exts])
    if len(files) == 0:
        raise FileNotFoundError(f'No audio found in {TEST_DIR}')

    # 3) 逐檔推論，取 Top-3
    results = {}
    for idx, path in enumerate(files, start=1):
        try:
            x_np = song_to_slices(path)
            song_probs = infer_one_song(net, device, x_np)  # [C]
            topk = min(3, len(song_probs))
            top_idx = np.argsort(-song_probs)[:topk]
            top_names = [class_names[i] for i in top_idx]
        except Exception as e:
            print(f'[warn] 推論失敗：{path.name} -> {e}')
            top_names = []

        key = f'{idx:03d}'  # "001", "002", ...
        results[key] = top_names

        if idx % 10 == 0 or idx == len(files):
            print(f'  {idx}/{len(files)} done...')

    # 4) 存 JSON
    (OUT_JSON.parent).mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print('[done] 寫入：', OUT_JSON)
    print(f'[time] {time.time()-t0:.1f}s for {len(files)} songs')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gid', type=int, default=0, help='GPU index')
    args = parser.parse_args()
    if torch.cuda.is_available():
        with torch.cuda.device(args.gid):
            main(gid=args.gid)
    else:
        print('[warn] CUDA 不可用，改用 CPU 推論。')
        main(gid=-1)
