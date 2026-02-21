# 进入环境
conda activate kohya_ss

# 执行训练

每次cancel了之后，GPU还在占用，运行`pkill -9 python`清理

## Training 1, very slow
```
accelerate launch --num_cpu_threads_per_process 8 \
  "./sd-scripts/flux_train_network.py" \
  --pretrained_model_name_or_path "./models/FLUX.1-dev/flux1-dev.safetensors" \
  --clip_l "./models/FLUX.1-dev/text_encoder/model.safetensors" \
  --t5xxl "./models/FLUX.1-dev/t5xxl_fp8_e4m3fn.safetensors" \
  --ae "./models/FLUX.1-dev/ae.safetensors" \
  --train_data_dir "./test_train" \
  --output_dir "./outputs" \
  --output_name "EWOLiyingZhao_Flux_v1" \
  --save_model_as safetensors \
  --sdpa \
  --mixed_precision bf16 \
  --network_module networks.lora_flux \
  --network_dim 32 \
  --network_alpha 16 \
  --resolution "512,512" \
  --train_batch_size 2 \
  --max_train_steps 2000 \
  --save_every_n_steps 500 \
  --learning_rate 1e-4 \
  --optimizer_type "AdamW8bit" \
  --cache_latents \
  --cache_latents_to_disk \
  --gradient_checkpointing \
  --save_precision bf16 \
  --caption_extension ".txt"
```

## Training 2 (训练坍缩，只能产生雪花图)

```
accelerate launch --num_cpu_threads_per_process 4 \
  "./sd-scripts/flux_train_network.py" \
  --pretrained_model_name_or_path "./models/FLUX.1-dev/flux1-dev.safetensors" \
  --clip_l "./models/FLUX.1-dev/text_encoder/model.safetensors" \
  --t5xxl "./models/FLUX.1-dev/t5xxl_fp8_e4m3fn.safetensors" \
  --ae "./models/FLUX.1-dev/ae.safetensors" \
  --train_data_dir "./test_train" \
  --output_dir "./outputs" \
  --output_name "EWOLiyingZhao_Flux_v1" \
  --sdpa --mixed_precision bf16 \
  --network_module networks.lora_flux \
  --network_dim 32 --network_alpha 16 \
  --resolution "512,512" \
  --train_batch_size 2 \
  --max_train_steps 2000 \
  --learning_rate 1e-4 \
  --optimizer_type "AdamW8bit" \
  --cache_latents \
  --cache_text_encoder_outputs \
  --gradient_checkpointing \
  --save_precision bf16 \
  --caption_extension ".txt"
```

## Training 3

可以训练，但最后出来权重是0.5或更高，全是雪花，如果是0.1，可以出图，但不像。设置权重

```python
print(f"🎨 注入 LoRA 权重并设置强度...")
# 1. 先加载权重
pipe.load_lora_weights(LORA_PATH, adapter_name="yusaku")

# 2. 设置极低强度 (0.1) 来排查是否权重溢出
# 如果 0.1 能出图，说明 LoRA 还能救；如果 0.1 还是雪花，说明权重彻底炸了。
pipe.set_adapters(["yusaku"], adapter_weights=[0.1]) 

print("⚡ 5090 正在以 0.1 强度进行降压推理...")
```

核心改动：
1. 分辨率务必改回 1024,1024
2. 降低学习率，防止再次坍缩 (推荐 5e-5 或 1e-4)
3. 开启显存优化，确保 32GB 稳跑

```
accelerate launch --num_cpu_threads_per_process 4 \
  "./sd-scripts/flux_train_network.py" \
  --pretrained_model_name_or_path "./models/FLUX.1-dev/flux1-dev.safetensors" \
  --clip_l "./models/FLUX.1-dev/text_encoder/model.safetensors" \
  --t5xxl "./models/FLUX.1-dev/t5xxl_fp8_e4m3fn.safetensors" \
  --ae "./models/FLUX.1-dev/ae.safetensors" \
  --train_data_dir "./test_train" \
  --output_dir "./outputs" \
  --output_name "EWOLiyingZhao_Flux_v2_1024" \
  --resolution "1024,1024" \
  --train_batch_size 1 \
  --max_train_steps 1500 \
  --save_every_n_steps 250 \
  --learning_rate 5e-5 \
  --network_dim 16 \
  --network_alpha 8 \
  --optimizer_type "AdamW8bit" \
  --mixed_precision bf16 \
  --sdpa \
  --cache_latents \
  --cache_text_encoder_outputs \
  --gradient_checkpointing
```

## Training 4

### 从0到1000 step（会出现雪花）
```
accelerate launch --num_cpu_threads_per_process 2 \
  "./sd-scripts/flux_train_network.py" \
  --pretrained_model_name_or_path "./models/FLUX.1-dev/flux1-dev.safetensors" \
  --clip_l "./models/FLUX.1-dev/text_encoder/model.safetensors" \
  --t5xxl "./models/FLUX.1-dev/t5xxl_fp8_e4m3fn.safetensors" \
  --ae "./models/FLUX.1-dev/ae.safetensors" \
  --network_module networks.lora_flux \
  --train_data_dir "./test_train" \
  --output_dir "./outputs" \
  --output_name "EWOLiyingZhao_Flux_v5_Success" \
  --resolution "1024,1024" \
  --network_dim 32 \
  --network_alpha 16 \
  --train_batch_size 1 \
  --max_train_steps 2500 \
  --save_every_n_steps 500 \
  --learning_rate 1e-4 \
  --optimizer_type "AdamW" \
  --mixed_precision bf16 \
  --sdpa \
  --cache_latents \
  --cache_text_encoder_outputs \
  --gradient_checkpointing \
  --save_precision bf16 \
  --lowram \
  --caption_extension ".txt"
```

### step 1000开始精修

```
accelerate launch --num_cpu_threads_per_process 2 \
  "./sd-scripts/flux_train_network.py" \
  --pretrained_model_name_or_path "./models/FLUX.1-dev/flux1-dev.safetensors" \
  --clip_l "./models/FLUX.1-dev/text_encoder/model.safetensors" \
  --t5xxl "./models/FLUX.1-dev/t5xxl_fp8_e4m3fn.safetensors" \
  --ae "./models/FLUX.1-dev/ae.safetensors" \
  --network_module networks.lora_flux \
  --network_weights "./outputs/EWOLiyingZhao_Flux_v5_Success-step00001000.safetensors" \
  --train_data_dir "./test_train" \
  --output_dir "./outputs" \
  --output_name "EWOLiyingZhao_Flux_v6_Refined" \
  --resolution "1024,1024" \
  --network_dim 32 \
  --network_alpha 16 \
  --train_batch_size 1 \
  --max_train_steps 600 \
  --save_every_n_steps 200 \
  --learning_rate 2e-5 \
  --optimizer_type "AdamW" \
  --mixed_precision bf16 \
  --sdpa \
  --cache_latents \
  --cache_text_encoder_outputs \
  --gradient_checkpointing \
  --save_precision bf16 \
  --lowram \
  --caption_extension ".txt"
```

## Training 5

```
accelerate launch --num_cpu_threads_per_process 2 \
  "./sd-scripts/flux_train_network.py" \
  --pretrained_model_name_or_path "./models/FLUX.1-dev/flux1-dev.safetensors" \
  --clip_l "./models/FLUX.1-dev/text_encoder/model.safetensors" \
  --t5xxl "./models/FLUX.1-dev/t5xxl_fp8_e4m3fn.safetensors" \
  --ae "./models/FLUX.1-dev/ae.safetensors" \
  --network_module networks.lora_flux \
  --train_data_dir "./test_train" \
  --output_dir "./outputs" \
  --output_name "EWOLiyingZhao_Flux_Final_v7" \
  --resolution "1024,1024" \
  --network_dim 16 \
  --network_alpha 1 \
  --train_batch_size 1 \
  --max_train_steps 1500 \
  --learning_rate 4e-5 \
  --optimizer_type "AdamW" \
  --mixed_precision bf16 \
  --sdpa \
  --cache_latents \
  --cache_text_encoder_outputs \
  --gradient_checkpointing \
  --save_precision bf16 \
  --lowram
  ```
  