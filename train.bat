:git pull origin main
venv\Scripts\pip install -e .
set WANDB_API_KEY=ce3f15d2269ab54962d4c3095ceb5e87d891219c
venv\Scripts\accelerate launch --mixed_precision=fp16 D:\GIT\F5-TTS\src\f5_tts\train\finetune_cli.py --exp_name F5TTS_Base --learning_rate 1e-05 --batch_size_per_gpu 1000 --batch_size_type frame --max_samples 64 --grad_accumulation_steps 30 --max_grad_norm 1 --epochs 10 --num_warmup_updates 380000 --save_per_updates 100000 --keep_last_n_checkpoints -1 --last_per_updates 10000 --dataset_name russian --finetune --tokenizer_path D:\GIT\F5-TTS\data\russian_custom\vocab.txt --tokenizer custom --log_samples --logger wandb --bnb_optimizer
