set CUDA_VISIBLE_DEVICES=0
python train_wgan.py --n_epochs 2 --batch_size 256 --ht_dim 3 --ecg_dim 50 --lr 0.00005 --n_cpu 2 --samples 1000000