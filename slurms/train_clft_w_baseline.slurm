#!/bin/bash 
#SBATCH --job-name=v1w-base                ### job name
#SBATCH -n 1
#SBATCH -p gpu                  ### use gpu
#SBATCH --gres=gpu:A100:1               ### specifying the GPU type
#SBATCH --constraint="A100-80"
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH -t 5-0                  ### time limit 
#SBATCH --mail-type=ALL
#SBATCH --account=project_autolab

## run job   
source venv/bin/activate 

python train.py --config config/waymo_hpc/config_1.json
python test.py --config config/waymo_hpc/config_1.json
python visualize.py --config config/waymo_hpc/config_1.json --upload
python3 benchmark.py --config config/waymo_hpc/config_1.json

python train.py --config config/waymo_hpc/config_2.json
python test.py --config config/waymo_hpc/config_2.json
python visualize.py --config config/waymo_hpc/config_2.json --upload
python3 benchmark.py --config config/waymo_hpc/config_2.json

python train.py --config config/waymo_hpc/config_3.json
python test.py --config config/waymo_hpc/config_3.json
python visualize.py --config config/waymo_hpc/config_3.json --upload
python3 benchmark.py --config config/waymo_hpc/config_3.json

python train.py --config config/waymo_hpc/config_4.json
python test.py --config config/waymo_hpc/config_4.json
python visualize.py --config config/waymo_hpc/config_4.json --upload
python3 benchmark.py --config config/waymo_hpc/config_4.json

python train.py --config config/waymo_hpc/config_5.json
python test.py --config config/waymo_hpc/config_5.json
python visualize.py --config config/waymo_hpc/config_5.json --upload
python3 benchmark.py --config config/waymo_hpc/config_5.json

python train.py --config config/waymo_hpc/config_6.json
python test.py --config config/waymo_hpc/config_6.json
python visualize.py --config config/waymo_hpc/config_6.json --upload
python3 benchmark.py --config config/waymo_hpc/config_6.json

python train.py --config config/waymo_hpc/config_7.json
python test.py --config config/waymo_hpc/config_7.json
python visualize.py --config config/waymo_hpc/config_7.json --upload
python3 benchmark.py --config config/waymo_hpc/config_7.json

python train.py --config config/waymo_hpc/config_8.json
python test.py --config config/waymo_hpc/config_8.json
python visualize.py --config config/waymo_hpc/config_8.json --upload
python3 benchmark.py --config config/waymo_hpc/config_8.json

python train.py --config config/waymo_hpc/config_9.json
python test.py --config config/waymo_hpc/config_9.json
python visualize.py --config config/waymo_hpc/config_9.json --upload
python3 benchmark.py --config config/waymo_hpc/config_9.json
