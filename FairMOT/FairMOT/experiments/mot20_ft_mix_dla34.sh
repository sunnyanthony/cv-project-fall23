cd src
python train.py mot --gan --exp_id mot20_ft_mix_dla34 --load_model '../models/fairmot_dla34.pth' --num_epochs 20 --lr_step '15' --data_cfg '../src/lib/cfg/mot20.json' --gpus '1' --batch_size=10
cd ..
