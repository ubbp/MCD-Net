CUDA_VISIBLE_DEVICES=0 python test.py \
    --swin_type base \
    --dataset refsegrs \
    --split test \
    --window12 \
    --img_size 480 \
    --num_tmem 1 \
    --resume model_best_refsegrs.pth \
    --save_dir ./your_save_dir