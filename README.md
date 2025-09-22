# py3_test_diffusion_proj

test. diffusion local 測試搭建. 生成圖片

# 建立虛擬環境

python -m venv selfenv

# 啟動虛擬環境

source selfenv/bin/activate # Mac/Linux
selfenv\Scripts\activate # Windows

# 安裝必要套件（Mac/Apple Silicon 會使用 MPS，不支援 CUDA）

1. pip install -U pip
   pip install torch torchvision diffusers transformers

2.

# 核心套件

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # CUDA 11.8

# 如果用 MPS（Mac M1/M2），用 CPU 或 MPS 版本：

# pip install torch torchvision torchaudio

pip install diffusers transformers accelerate safetensors
pip install Pillow scipy ftfy

# 記憶體/速度優化

pip install xformers # 支援 memory efficient attention

# 進階使用

pip install matplotlib ipywidgets # Notebook 可視化

# 生成圖片（自動選擇 CUDA/MPS/CPU）

python sd_generate.py --prompt "a photo of a cute cat" --steps 25 --out out.png

# 若在 macOS 上，想允許 MPS 未支援運算回退到 CPU（可選）

export PYTORCH_ENABLE_MPS_FALLBACK=1

# 退出虛擬環境

deactivate
