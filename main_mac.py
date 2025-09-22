from __future__ import print_function
import os
import argparse
import datetime
import torch
from pathlib import Path
from diffusers import StableDiffusionPipeline


def main():
    print("-Start.")
    parser = argparse.ArgumentParser(description="使用 Stable Diffusion 產生圖片")
    parser.add_argument("--prompt", type=str, default="a photo of a cute cat", help="文字提示（要生成的內容）")
    parser.add_argument("--steps", type=int, default=25, help="推論步數（越多越細緻，速度越慢）")
    args = parser.parse_args()

    device = 'mps'  
    dtype = torch.float16
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    pipe = StableDiffusionPipeline.from_pretrained(
			"runwayml/stable-diffusion-v1-5",
			safety_checker=None,
			torch_dtype=dtype,
			low_cpu_mem_usage=True
		)
    pipe = pipe.to(device)
    prompt = args.prompt
    image = pipe(prompt,num_inference_steps= args.steps, guidance_scale=7.5).images[0]

    file =f"output/P-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.png"
    out_path = Path(file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    print(f"--SUCCESS {file}")

if __name__ == "__main__":
    main()