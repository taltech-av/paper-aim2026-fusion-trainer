#!/bin/bash 
#SBATCH --job-name=z-base                ### job name
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

python train_clft.py --config config/zod/clft/config_1.json
python test_clft.py --config config/zod/clft/config_1.json
python visualize_clft.py --config config/zod/clft/config_1.json --upload
python benchmark.py --config config/zod/clft/config_1.json

python train_clft.py --config config/zod/clft/config_2.json
python test_clft.py --config config/zod/clft/config_2.json
python visualize_clft.py --config config/zod/clft/config_2.json --upload
python benchmark.py --config config/zod/clft/config_2.json

python train_clft.py --config config/zod/clft/config_3.json
python test_clft.py --config config/zod/clft/config_3.json
python visualize_clft.py --config config/zod/clft/config_3.json --upload
python benchmark.py --config config/zod/clft/config_3.json

python train_clft.py --config config/zod/clft/config_4.json
python test_clft.py --config config/zod/clft/config_4.json
python visualize_clft.py --config config/zod/clft/config_4.json --upload
python benchmark.py --config config/zod/clft/config_4.json

python train_clft.py --config config/zod/clft/config_5.json
python test_clft.py --config config/zod/clft/config_5.json
python visualize_clft.py --config config/zod/clft/config_5.json --upload
python benchmark.py --config config/zod/clft/config_5.json

python train_clft.py --config config/zod/clft/config_6.json
python test_clft.py --config config/zod/clft/config_6.json
python visualize_clft.py --config config/zod/clft/config_6.json --upload
python benchmark.py --config config/zod/clft/config_6.json

python train_clft.py --config config/zod/clft/config_7.json
python test_clft.py --config config/zod/clft/config_7.json
python visualize_clft.py --config config/zod/clft/config_7.json --upload
python benchmark.py --config config/zod/clft/config_7.json

python train_clft.py --config config/zod/clft/config_8.json
python test_clft.py --config config/zod/clft/config_8.json
python visualize_clft.py --config config/zod/clft/config_8.json --upload
python benchmark.py --config config/zod/clft/config_8.json

python train_clft.py --config config/zod/clft/config_9.json
python test_clft.py --config config/zod/clft/config_9.json
python visualize_clft.py --config config/zod/clft/config_9.json --upload
python benchmark.py --config config/zod/clft/config_9.json
