from __future__ import print_function
import os
import argparse
import datetime
import torch
from pathlib import Path
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler


def main():
    print("-Start.")
    parser = argparse.ArgumentParser(description="使用 Stable Diffusion 產生圖片")
    parser.add_argument("--prompt", type=str, default="a photo of a cute cat", help="文字提示（要生成的內容）")
    parser.add_argument("--neg", type=str, default=None, help="negative prompt")
    parser.add_argument("--steps", type=int, default=25, help="推論步數（越多越細緻，速度越慢）")
    args = parser.parse_args()

    device = 'mps'  
    dtype = torch.float16
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "model", "chilloutmix_NiPrunedFp16.safetensors"))
    pipe = StableDiffusionPipeline.from_single_file(
			model_path,
			safety_checker=None,
			torch_dtype=dtype,
			low_cpu_mem_usage=False
		)
    pipe = pipe.to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    neg = args.neg or "low quality, blurry, cartoon, bad anatomy, extra limbs, fused fingers, missing fingers, lowres, blurry, deformed face, unrealistic, cartoon, bad hands, poorly drawn hands, mutated hands, bad proportions, jpeg artifacts, blurry, out of focus, extra fingers, missing fingers, disfigured, mutated, bad anatomy, unrealistic, low quality, low resolution, watermark, text, signature, jpeg artifacts"
    image = pipe(height=768, width=512, prompt=args.prompt, negative_prompt=neg, num_inference_steps= args.steps, guidance_scale=7.5).images[0]

    file =f"output/P-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.png"
    out_path = Path(file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    print(f"--SUCCESS {file}")

if __name__ == "__main__":
    main()