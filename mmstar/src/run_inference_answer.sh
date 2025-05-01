CUDA_VISIBLE_DEVICES=0,1,2,3 python /mnt/mm-cot/alpacabase_inference.py \
    --data_root /mnt/mm-cot/mmstar_data \
    --caption_file /mnt/mm-cot/mmstar_data/instruct_captions.json \
    --model declare-lab/flan-alpaca-base \
    --user_msg answer --img_type vit \
    --bs 2 --eval_bs 2 --epoch 2 --lr 5e-5 --output_len 64 \
    --use_caption --use_generate --prompt_format QCMG-A \
    --output_dir /mnt/mm-cot/experiments   \     
    
          
