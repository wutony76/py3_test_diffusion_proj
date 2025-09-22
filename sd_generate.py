# 說明: 使用 Hugging Face Diffusers 的 Stable Diffusion 依據文字提示生成圖片
# 功能: 自動偵測裝置 (CUDA/MPS/CPU)、可調步數與引導強度、支援指定模型與輸出路徑

import argparse
import os
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline


def select_device() -> str:
	"""選擇最佳可用裝置順序：CUDA > MPS > CPU。"""
	# 優先使用 NVIDIA CUDA (最快，若有支援的顯卡)
	if torch.cuda.is_available():
		return "cuda"
	# 其次使用 Apple Silicon 的 MPS 後端 (適用於 macOS 12.3+ 的 Apple GPU)
	if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
		return "mps"
	# 最後退回 CPU
	return "cpu"


def main() -> None:
	# 解析命令列參數
	parser = argparse.ArgumentParser(description="使用 Stable Diffusion 產生圖片（自動選擇 CUDA/MPS/CPU）")
	parser.add_argument("--prompt", type=str, default="a photo of a cute cat", help="文字提示（要生成的內容）")
	parser.add_argument("--steps", type=int, default=25, help="推論步數（越多越細緻，速度越慢）")
	parser.add_argument("--guidance", type=float, default=7.5, help="Classifier-free guidance 強度（越高越貼近提示）")
	parser.add_argument("--model", type=str, default="runwayml/stable-diffusion-v1-5", help="模型 ID 或本地路徑")
	parser.add_argument("--out", type=str, default="out.png", help="輸出圖片檔案路徑")
	args = parser.parse_args()

	# 選擇運算裝置
	device = select_device()
	print(f"Using device: {device}")

	# MPS（Apple GPU）上有些運算尚未實作，開啟 fallback 可自動回退到 CPU
	if device == "mps":
		os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

	# 在 GPU 類裝置（CUDA/MPS）上使用 float16 可節省記憶體；CPU 上維持 float32
	dtype = torch.float16 if device in ("cuda", "mps") else torch.float32

	# 載入 Stable Diffusion 管線
	print(f"Loading pipeline: {args.model} (dtype={dtype}, device={device})")
	pipe = StableDiffusionPipeline.from_pretrained(
		args.model,
		safety_checker=None,  # 移除 NSFW 濾鏡
		torch_dtype=dtype,
	)
	# 將模型移動到對應裝置
	pipe = pipe.to(device)

	# 依提示文字產生單張圖片
	image = pipe(
		args.prompt,
		num_inference_steps=args.steps,
		guidance_scale=args.guidance,
	).images[0]

	# 確保輸出資料夾存在並儲存圖片
	out_path = Path(args.out)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	image.save(out_path)
	print(f"Saved image to: {out_path.resolve()}")


if __name__ == "__main__":
	main() 