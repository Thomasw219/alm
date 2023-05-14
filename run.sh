export ENV_NAME='HalfCheetah-v3'
python train.py id=$ENV_NAME seed=0 device=cuda:0 &
python train.py id=$ENV_NAME seed=1 device=cuda:0 &
python train.py id=$ENV_NAME seed=2 device=cuda:0 &
python train.py id=$ENV_NAME seed=3 device=cuda:1 &
python train.py id=$ENV_NAME seed=4 device=cuda:1 &