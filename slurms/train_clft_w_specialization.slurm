#!/bin/bash 
#SBATCH --job-name=v1w-spec                ### job name
#SBATCH -n 1
#SBATCH -p gpu                  ### use gpu
#SBATCH --gres=gpu:A100:1               ### specifying the GPU type
#SBATCH --constraint="A100-80"
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH -t 5-0                  ### time limit 
#SBATCH --mail-type=ALL
#SBATCH --account=project_autolab

## run job
source venv/bin/activate 

python train.py --config config/waymo/clft/human_sign.json
python test.py --config config/waymo/clft/human_sign.json
python visualize.py --config config/waymo/clft/human_sign.json --upload
python benchmark.py --config config/waymo/clft/human_sign.json

python train.py --config config/waymo/clft/human_specialization.json
python test.py --config config/waymo/clft/human_specialization.json
python visualize.py --config config/waymo/clft/human_specialization.json --upload
python benchmark.py --config config/waymo/clft/human_specialization.json

python train.py --config config/waymo/clft/human_vehicle.json
python test.py --config config/waymo/clft/human_vehicle.json
python visualize.py --config config/waymo/clft/human_vehicle.json --upload
python benchmark.py --config config/waymo/clft/human_vehicle.json

python train.py --config config/waymo/clft/sign_specialization.json
python test.py --config config/waymo/clft/sign_specialization.json
python visualize.py --config config/waymo/clft/sign_specialization.json --upload
python benchmark.py --config config/waymo/clft/sign_specialization.json

python train.py --config config/waymo/clft/vehicle_sign.json
python test.py --config config/waymo/clft/vehicle_sign.json
python visualize.py --config config/waymo/clft/vehicle_sign.json --upload
python benchmark.py --config config/waymo/clft/vehicle_sign.json

python train.py --config config/waymo/clft/vehicle_specialization.json
python test.py --config config/waymo/clft/vehicle_specialization.json
python visualize.py --config config/waymo/clft/vehicle_specialization.json --upload
python benchmark.py --config config/waymo/clft/vehicle_specialization.json
