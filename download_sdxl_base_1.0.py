# 保存为 download_sdxl.py
from diffusers import StableDiffusionXLPipeline
import torch

model_id = "stabilityai/stable-diffusion-xl-base-1.0"

print(f"🚀 正在下载 SDXL Base 1.0 到本地模型目录...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    variant="fp16", 
    use_safetensors=True,
    cache_dir="./models/SDXL" # 指定你想要的存储位置
)
print("✅ 下载完成！")

