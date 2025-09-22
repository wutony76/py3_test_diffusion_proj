# py3_test_diffusion_proj

使用 Hugging Face Diffusers 的 Stable Diffusion 依文字提示生成圖片，支援自動選擇裝置（CUDA / MPS / CPU）。

---

## 環境準備

### 建立虛擬環境

```bash
python -m venv selfenv
```

### 啟動虛擬環境

- macOS / Linux

```bash
source selfenv/bin/activate
```

- Windows

```bash
selfenv\Scripts\activate
```

---

## 安裝必要套件

先更新 pip：

```bash
pip install -U pip
```

### 1) PyTorch（依裝置選擇其一）

- NVIDIA CUDA（範例使用 CUDA 11.8）：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

- Apple Silicon（M1/M2/M3，使用 MPS）或僅 CPU：

```bash
pip install torch torchvision torchaudio
```

### 2) Diffusers 與相關套件

```bash
pip install diffusers transformers safetensors
pip install Pillow scipy ftfy
```

可選（進階／開發工具）：

```bash
# 僅建議 CUDA 使用者安裝（部分平台不提供）：
pip install xformers

# Notebook 可視化：
pip install matplotlib ipywidgets
```

---

## 使用方式

本專案提供兩種執行方式：

### 方式 A：macOS 快速腳本（`main_mac.py`）

僅測試用，固定使用 `MPS` 與模型 `runwayml/stable-diffusion-v1-5`。

```bash
selfenv/bin/python main_mac.py --prompt "a photo of a cute cat" --steps 25
```

輸出檔案會自動以時間命名，例如：`P-20250922171850.png`。

### 方式 B：跨平台 CLI（`sd_generate.py`）

會自動偵測最佳裝置（CUDA > MPS > CPU），並提供更多參數。

顯示說明：

```bash
python sd_generate.py -h
```

基本用法：

```bash
python sd_generate.py \
  --prompt "a photo of a cute cat" \
  --steps 25 \
  --guidance 7.5 \
  --out out.png
```

指定模型（可用本地路徑或 Hugging Face Hub ID）：

```bash
python sd_generate.py \
  --prompt "ultra-detailed portrait, 4k" \
  --model runwayml/stable-diffusion-v1-5 \
  --steps 30 \
  --guidance 7.0 \
  --out outputs/portrait.png
```

首次執行會自動下載模型權重，依網路速度與磁碟空間而定。

---

## macOS 使用者小提示（MPS 回退）

某些運算在 MPS 尚未實作時可自動回退 CPU，建議在 macOS 上開啟：

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

---

## 離開虛擬環境

```bash
deactivate
```
