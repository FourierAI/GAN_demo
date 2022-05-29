set CUDA_VISIBLE_DEVICES=0
python train_wgan.py --n_epochs 1 --batch_size 256 --ht_dim 1 --ecg_dim 50 --lr 0.00005 --n_cpu 3 --samples 3000000