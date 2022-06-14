# test
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --use_env  --master_port 29534 main.py  --model vit_deit_tiny_patch16_224 --input-size 224 --batch-size 16 --data-path /media/DATASET/ImageNet --data-set IMNET --output_dir ./log/imagenet1000_small_224_test --dist-eval --wandb_project transformer_reparam --wandb_name test 
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --use_env  --master_port 29534 main.py  --model vit_deit_tiny_patch16_224 --input-size 224 --batch-size 16 --data-path /media/DATASET/ImageNet --data-set IMNET --output_dir ./log/imagenet1000_small_224_test --dist-eval --wandb_project transformer_reparam --wandb_name test --autoresume 

# test one epoch
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --use_env  --master_port 29534 main.py  --model vit_deit_tiny_patch16_224 --input-size 224 --batch-size 256 --data-path /media/DATASET/ImageNet --data-set IMNET --output_dir ./log/imagenet1000_small_224_test --dist-eval --wandb_project transformer_reparam --wandb_name test 
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --use_env  --master_port 29534 main.py  --model vit_deit_tiny_patch16_224 --input-size 224 --batch-size 256 --data-path /media/DATASET/ImageNet --data-set IMNET --output_dir ./log/imagenet1000_small_224_test --dist-eval --wandb_project transformer_reparam --wandb_name test --autoresume