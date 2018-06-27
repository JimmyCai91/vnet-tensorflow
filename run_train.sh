#patch_size W and H
#patch_layer deepth
path_to_volume_label_folder='/home/jinzhengc/Data/Task03_Liver'
path_to_log_dir='checkpoints/Task03_Liver_VNet/log'
mkdir -p ${path_to_log_dir}
path_to_ckpt='checkpoints/Task03_Liver_VNet/ckpt'
mkdir -p ${path_to_ckpt}
path_to_model='checkpoints/Task03_Liver_VNet/model'
mkdir -p ${path_to_model}

export CUDA_VISIBLE_DEVICES=0
python train.py \
    --data_dir ${path_to_volume_label_folder} \
    --batch_size 1 \
    --patch_size 128 \
    --patch_layer 128 \
    --epochs 50 \
    --log_dir ${path_to_log_dir} \
    --init_learning_rate 0.000005 \
    --decay_factor 0.01 \
    --decay_steps 10 \
    --display_step 10 \
    --save_interval 1 \
    --checkpoint_dir ${path_to_ckpt} \
    --model_dir ${path_to_model} \
    --restore_training \
    --drop_ratio 0.01 \
    --min_pixel 4096 \
    --shuffle_buffer_size 5
