accelerate launch  --mixed_precision bf16 --num_cpu_threads_per_process 6 flux_train_network.py --pretrained_model_name_or_path "D:/SDXL/webui_forge_cu121_torch231/webui/models/Stable-diffusion/flux1-dev.safetensors" --clip_l "D:/SDXL/webui_forge_cu121_torch231/webui/models/VAE/clip_l.safetensors" --t5xxl "D:/SDXL/webui_forge_cu121_torch231/webui/models/VAE/t5xxl_fp16.safetensors" --ae "D:/SDXL/webui_forge_cu121_torch231/webui/models/VAE/ae.safetensors" --cache_latents --cache_latents_to_disk --save_model_as safetensors --sdpa --persistent_data_loader_workers --max_data_loader_n_workers 6 --seed 42 --gradient_checkpointing --mixed_precision bf16 --save_precision bf16 --network_module networks.lora_flux --network_dim 16 --learning_rate 5e-4 --network_train_unet_only --cache_text_encoder_outputs --cache_text_encoder_outputs_to_disk --fp8_base --resolution="512,512" --save_every_n_steps="100" --train_data_dir="F:/train/data/sakuranomiya_maika" --output_dir "D:/SDXL/webui_forge_cu121_torch231/webui/models/Lora/sakuranomiya_maika" --logging_dir "F:/sakuranomiya_maika" --output_name sakuranomiya_maika_18 --timestep_sampling shift --discrete_flow_shift 3.1582 --model_prediction_type raw --guidance_scale 1.0 --loss_type l2 --optimizer_type came --optimizer_args "betas=0.9,0.999,0.9999" "weight_decay=0.01" --max_train_steps="1000" --enable_bucket --caption_extension=".txt" --train_batch_size=4 --apply_t5_attn_mask --noise_offset 0.05 --lr_scheduler "REX" --lr_warmup_steps 100 --split_mode --network_args "train_blocks=single"

:: 先按照GUI的標準方式安裝 然後自己修改FLUX.bat的參數
:: 用.\venv\Scripts\activate 進入venv再執行FLUX.bat

:: 這個fork增加了 CAME優化器和REX排程 你可以簡單當成是更好的adafactor和更好的cosine

:: 這個設定是FOR 12G
:: 16G VRAM 拿掉最後面的 --split_mode --network_args "train_blocks=single"
:: 24G VRAM 再加上 --highvram

:: flux1-dev.safetensors ae.safetensors
:: 請至 https://huggingface.co/black-forest-labs/FLUX.1-dev/tree/main 下載

:: t5xxl_fp16.safetensors clip_l.safetensors 
:: 請至 https://huggingface.co/stabilityai/stable-diffusion-3-medium/tree/main/text_encoders 下載