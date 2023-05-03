# python train.py --batch_size=64 --max_epoch=100 --eval_interval=10 --save_interval=5 --gradient_accumulation_interval=16 --output_dir=/content/drive/MyDrive/tmp/checkpoints --model_size=LLAMA --with_lr_scheduler --dataset_path= --save_cache 

# python train.py --batch_size=64 --max_epoch=100 --eval_interval=10 --save_interval=5 --gradient_accumulation_interval=16 --output_dir=/content/drive/MyDrive/tmp/checkpoints/ --model_size=LLAMA --with_lr_scheduler --from_cache --cache_directory=/content/drive/MyDrive/dataset_cache.tar.gz

torchrun train.py --batch_size=128 --max_epoch=100 --eval_interval=10 --save_interval=5 --gradient_accumulation_interval=2 --output_dir=/content/drive/MyDrive/tmp/checkpoints/ --model_size=LLAMA --with_lr_scheduler --from_cache --cache_directory=/content/drive/MyDrive/corpus.tar.gz
