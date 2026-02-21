# 进入环境
conda activate kohya_ss

# 执行训练

每次cancel了之后，GPU还在占用，运行`pkill -9 python`清理

## Train 1 (成功但质量差)

```
accelerate launch --num_cpu_threads_per_process 2 \
  "./sd-scripts/train_network.py" \
  --pretrained_model_name_or_path "./models/v1-5-pruned.safetensors" \
  --train_data_dir "./test_train" \
  --output_dir "./outputs" \
  --output_name "SD15_LiyingZhao_v1" \
  --resolution "512,512" \
  --network_module networks.lora \
  --network_dim 32 \
  --network_alpha 16 \
  --train_batch_size 4 \
  --max_train_steps 2000 \
  --save_every_n_steps 500 \
  --learning_rate 1e-4 \
  --optimizer_type "AdamW8bit" \
  --mixed_precision bf16 \
  --xformers \
  --cache_latents \
  --gradient_checkpointing \
  --save_precision fp16 \
  --caption_extension ".txt"
```

## Train 2 (失败)

```
accelerate launch --num_cpu_threads_per_process 2 \
  "./sd-scripts/sdxl_train_network.py" \
  --pretrained_model_name_or_path "./models/sd_xl_base_1.0.safetensors" \
  --train_data_dir "./test_train" \
  --output_dir "./outputs" \
  --output_name "SDXL_LYZ_Success_v1" \
  --resolution "1024,1024" \
  --network_module networks.lora \
  --network_dim 32 \
  --network_alpha 16 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --max_train_steps 1500 \
  --learning_rate 1e-4 \
  --optimizer_type "AdamW8bit" \
  --mixed_precision bf16 \
  --save_precision fp16 \
  --caption_extension ".txt" \
  --no_half_vae \
  --noise_offset 0.1 \
  --min_snr_gamma 5.0 \
  --gradient_checkpointing
```

## Train 3 (DreamBooth)

```
# 1. 物理清理缓存（虽然 zsh 报错找不到，但手动清理 npz 依然重要）
rm test_train/*.npz 2>/dev/null

# 2. 运行“稳定性优先版”
accelerate launch --num_cpu_threads_per_process 2 \
  "./sd-scripts/sdxl_train.py" \
  --pretrained_model_name_or_path "./models/sd_xl_base_1.0.safetensors" \
  --train_data_dir "./test_train" \
  --output_dir "./outputs" \
  --output_name "SDXL_DB_LYZ_Safe" \
  --resolution "1024,1024" \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --max_train_steps 1500 \
  --save_every_n_steps 300 \
  --learning_rate 1e-6 \
  --optimizer_type "AdamW8bit" \
  --mixed_precision bf16 \
  --save_precision fp16 \
  --no_half_vae \
  --noise_offset 0.1 \
  --gradient_checkpointing \
  --sdpa \
  --cache_latents \
  --cache_text_encoder_outputs \
  --highvram
```

### 从第300步继续

```
accelerate launch --num_cpu_threads_per_process 2 \
  "./sd-scripts/sdxl_train.py" \
  --pretrained_model_name_or_path "./outputs/SDXL_DB_LYZ_Safe-step00000300.ckpt" \
  --train_data_dir "./test_train" \
  --output_dir "./outputs" \
  --output_name "SDXL_DB_LYZ_Resume" \
  --resolution "1024,1024" \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --max_train_steps 1500 \
  --save_every_n_steps 300 \
  --learning_rate 1e-6 \
  --optimizer_type "AdamW8bit" \
  --mixed_precision bf16 \
  --save_precision fp16 \
  --no_half_vae \
  --noise_offset 0.1 \
  --gradient_checkpointing \
  --sdpa \
  --cache_latents \
  --cache_text_encoder_outputs \
  --highvram
```

### 抽取LoRA

```
python "./sd-scripts/networks/extract_lora_from_models.py" \
  --model_tuned "./outputs/SDXL_DB_LYZ_Resume.ckpt" \
  --model_org "./models/sd_xl_base_1.0.safetensors" \
  --save_to "./outputs/SDXL_DB_LYZ_Resume_Extracted_LoRA.safetensors" \
  --dim 64 \
  --sdxl \
  --device cuda
```

## Train 4 (DreamBooth with different base model gonzales20260217Dmd.dzxk.safetensors)

初始训练
```
accelerate launch --num_cpu_threads_per_process 2 \
  "./sd-scripts/sdxl_train.py" \
  --pretrained_model_name_or_path "./models/gonzales20260217Dmd.dzxk.safetensors" \
  --train_data_dir "./test_train" \
  --output_dir "./outputs" \
  --output_name "SDXL_gonzales20260217Dmd" \
  --resolution "1024,1024" \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --max_train_steps 1500 \
  --save_every_n_steps 300 \
  --learning_rate 1e-6 \
  --optimizer_type "AdamW8bit" \
  --mixed_precision bf16 \
  --save_precision fp16 \
  --no_half_vae \
  --noise_offset 0.1 \
  --gradient_checkpointing \
  --sdpa \
  --cache_latents \
  --cache_text_encoder_outputs \
  --highvram
```

从第900步开始
```
accelerate launch --num_cpu_threads_per_process 2 \
  "./sd-scripts/sdxl_train.py" \
  --pretrained_model_name_or_path "./outputs/SDXL_gonzales20260217Dmd-step00000900.ckpt" \
  --train_data_dir "./test_train" \
  --output_dir "./outputs" \
  --output_name "SDXL_gonzales20260217Dmd_resume900" \
  --resolution "1024,1024" \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --max_train_steps 600 \
  --save_every_n_steps 300 \
  --learning_rate 1e-6 \
  --optimizer_type "AdamW8bit" \
  --mixed_precision bf16 \
  --save_precision fp16 \
  --no_half_vae \
  --noise_offset 0.1 \
  --gradient_checkpointing \
  --sdpa \
  --cache_latents \
  --cache_text_encoder_outputs \
  --highvram
```

### 抽取LoRA

```
python "./sd-scripts/networks/extract_lora_from_models.py" \
  --model_tuned "./outputs/SDXL_gonzales20260217Dmd-step00001500.ckpt" \
  --model_org "./models/gonzales20260217Dmd.dzxk.safetensors" \
  --save_to "./outputs/SDXL_gonzales20260217Dmd-step00001500_Extracted_LoRA.safetensors" \
  --dim 64 \
  --sdxl \
  --device cuda
```
