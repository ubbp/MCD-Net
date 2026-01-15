CUDA_VISIBLE_DEVICES=0 python train.py \
    --dataset refsegrs  \
    --model_id MCDNet  \
    --epochs 100  \
    --lr 5e-5   \
    --num_tmem 1 \
    --output-dir ./your_output_path \
    --refer_data_root ./your_data_path  