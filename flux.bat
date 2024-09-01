
:: 先按照GUI的標準方式安裝 然後自己修改FLUX.bat的參數
:: 用.\venv\Scripts\activate 進入venv再執行FLUX.bat

:: 這個fork增加了 CAME優化器和REX排程 你可以簡單當成是更好的adafactor和更好的cosine

:: 這個設定是FOR 12G
:: 16G VRAM 拿掉最後面的 --split_mode --network_args "train_blocks=single"
:: 24G VRAM 再加上 --highvram

:: flux1-dev.safetensors 
:: ae.safetensors
:: 請至 https://huggingface.co/black-forest-labs/FLUX.1-dev/tree/main 下載

:: t5xxl_fp16.safetensors
:: clip_l.safetensors 
:: 請至 https://huggingface.co/stabilityai/stable-diffusion-3-medium/tree/main/text_encoders 下載


@echo off

:: 設定目錄變數
set TRAIN_DATA_DIR=F:/train/data/sakuranomiya_maika
set OUTPUT_DIR=D:/SDXL/webui_forge_cu121_torch231/webui/models/Lora/sakuranomiya_maika
set LOGGING_DIR=D:/SDXL/webui_forge_cu121_torch231/webui/models/Lora/sakuranomiya_maika
set OUTPUT_NAME=sakuranomiya_maika_33

:: 設定步數
set TRAIN_BATCH_SIZE=4
set MAX_TRAIN_STEPS=2000
set /A WARMUP_STEPS=%MAX_TRAIN_STEPS%/10

:: 設定模型變數
set PRETRAINED_MODEL=D:/SDXL/webui_forge_cu121_torch231/webui/models/Stable-diffusion/flux1-dev.safetensors
set VAE_MODEL_DIR=D:/SDXL/webui_forge_cu121_torch231/webui/models/VAE
set CLIP_L_MODEL=D:/SDXL/webui_forge_cu121_torch231/webui/models/VAE/clip_l.safetensors
set T5XXL_MODEL=D:/SDXL/webui_forge_cu121_torch231/webui/models/VAE/t5xxl_fp16.safetensors

accelerate launch  --mixed_precision bf16 --num_cpu_threads_per_process 6 flux_train_network.py --pretrained_model_name_or_path "%PRETRAINED_MODEL%" --clip_l "%CLIP_L_MODEL%" --t5xxl "%T5XXL_MODEL%" --ae "%VAE_MODEL_DIR%" --save_model_as safetensors --sdpa --persistent_data_loader_workers --max_data_loader_n_workers 6 --seed 42 --gradient_checkpointing --mixed_precision bf16 --save_precision bf16 --network_module networks.lora_flux --network_dim 16 --network_alpha 1 --learning_rate 1e-3 --network_train_unet_only --apply_t5_attn_mask --cache_text_encoder_outputs --cache_text_encoder_outputs_to_disk --fp8_base --resolution="512,512" --save_every_n_steps="100"  --train_data_dir "%TRAIN_DATA_DIR%" --output_dir "%OUTPUT_DIR%" --logging_dir "%LOGGING_DIR%" --output_name %OUTPUT_NAME%  --timestep_sampling flux_shift --model_prediction_type raw --guidance_scale 1.0 --loss_type l2  --optimizer_type adamwschedulefree --optimizer_args "weight_decay=0.1" "warmup_steps=%WARMUP_STEPS%" --max_train_steps="%MAX_TRAIN_STEPS%" --enable_bucket --caption_extension=".txt" --train_batch_size=%TRAIN_BATCH_SIZE% --split_mode --network_args "train_blocks=single"

tensorboard --logdir="%LOGGING_DIR%"
