# Flux.1 LoRA 训练全记录与避坑指南 (RTX 5090 & WSL2)

## 0. 前言
本记录汇总了在 RTX 5090 (32GB VRAM) 环境下，通过 WSL2 训练 Flux.1-dev 模型时遇到的所有底层资源竞争、环境限制及模型坍缩问题的解决方案。

---

## 1. 成功指令 (1024x1024 稳定版)
该配置在 **WSL2 (48GB RAM)** 与 **RTX 5090 (32GB VRAM)** 环境下验证通过。

```bash
accelerate launch --num_cpu_threads_per_process 2 \
  "./sd-scripts/flux_train_network.py" \
  --pretrained_model_name_or_path "./models/FLUX.1-dev/flux1-dev.safetensors" \
  --clip_l "./models/FLUX.1-dev/text_encoder/model.safetensors" \
  --t5xxl "./models/FLUX.1-dev/t5xxl_fp8_e4m3fn.safetensors" \
  --ae "./models/FLUX.1-dev/ae.safetensors" \
  --network_module networks.lora_flux \
  --train_data_dir "./test_train" \
  --output_dir "./outputs" \
  --output_name "EWOLiyingZhao_Flux_v2_1024" \
  --resolution "1024,1024" \
  --train_batch_size 1 \
  --max_train_steps 1500 \
  --save_every_n_steps 250 \
  --learning_rate 5e-5 \
  --optimizer_type "AdamW8bit" \
  --mixed_precision bf16 \
  --sdpa \
  --cache_latents \
  --cache_text_encoder_outputs \
  --gradient_checkpointing \
  --save_precision bf16 \
  --network_dim 8 \
  --network_alpha 4 \
  --lowram \
  --caption_extension ".txt"
```

## 2. 核心问题回顾与故障排查 (Troubleshooting)
| 现象 |	根本原因 |	最终对策 |
|--|--|--|
| 训练速度 443s/it | WSL2 内存瓶颈。默认分配不足导致系统频繁触发 Swap 读写硬盘。 | 修改 .wslconfig 将分配内存提至 48GB。 |
| 彩色噪点/花屏图 | 模型坍缩。在 512 分辨率下训练导致梯度爆炸，权重数值溢出。 | 强制回归 1024x1024 原生分辨率训练。|
| RuntimeError (Bias/Input mismatch) | VAE 精度冲突。手动将 VAE 转为 FP32 导致与 BF16 输入不匹配。 | 全链路统一使用 bf16，并使用 enable_model_cpu_offload。|
| FileNotFound / AttributeError | 参数缺失。漏掉 --network_module 或指向了错误的子目录。 | 明确指定 networks.lora_flux 模块。|
| Process exited with code 1 | 瞬时显存溢出。1024 模式下多线程数据加载撑爆了 32GB 显存。 | 开启 --lowram 并将 CPU 线程降至 2。|

## 3. 核心调优要点 (Developer Notes)
### 为什么 512 分辨率会“花屏”？
Flux 的 Flow Matching 架构是基于 1024 像素的全局分布设计的。低分辨率训练会导致流匹配的梯度计算产生严重偏差，使 LoRA 学习到的数值溢出（Exploding Weights）。在推理时，这些溢出权重会把随机噪声放大成无效的彩色斑点，导致 VAE 解码失败。

### 显存控制策略 (VRAM Management)
虽然 5090 算力顶尖，但 32GB 显存面对 Flux (12B 参数) 依然捉襟见肘：

* 训练级优化：必须开启 gradient_checkpointing 和 cache_text_encoder_outputs 以降低冗余计算占用的 Buffer。

* 推理级优化：必须使用 pipe.enable_model_cpu_offload()，它能将暂不计算的组件（如 T5）移出显存。

### WSL2 系统配置 (.wslconfig)
对于 Windows 64GB 宿主机，必须在 %UserProfile%\.wslconfig 设置以防止 OOM：

```
[wsl2]
memory=48GB
swap=32GB
pageReporting=true
```

## 4. 验证推理脚本 (Python)

```python
import torch
from diffusers import FluxPipeline

# 1. 严格以 bfloat16 加载整个管道，发挥 5090 硬件优势
pipe = FluxPipeline.from_pretrained("./models/FLUX.1-dev", torch_dtype=torch.bfloat16)

# 2. 核心：动态调度显存，防止单卡 32GB 环境死锁
pipe.enable_model_cpu_offload() 

# 3. 载入 LoRA
pipe.load_lora_weights("./outputs/EWOLiyingZhao_Flux_v2_1024.safetensors")

# 4. 执行推理
with torch.inference_mode():
    image = pipe(
        prompt="EWOLiyingZhao with long hair, smiling, blue shirt, 8k resolution",
        height=1024, width=1024,
        guidance_scale=3.5, 
        num_inference_steps=28
    ).images[0]

image.save("liying_zhao_final.png")
```
