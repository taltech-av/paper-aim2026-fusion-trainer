#!/bin/bash 
#SBATCH --job-name=mask-z                ### job name
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

python train_maskformer.py --config config/zod/maskformer/config_1.json
python test_maskformer.py --config config/zod/maskformer/config_1.json
python visualize_maskformer.py --config config/zod/maskformer/config_1.json --upload
python benchmark_maskformer.py --config config/zod/maskformer/config_1.json

python train_maskformer.py --config config/zod/maskformer/config_2.json
python test_maskformer.py --config config/zod/maskformer/config_2.json
python visualize_maskformer.py --config config/zod/maskformer/config_2.json --upload
python benchmark_maskformer.py --config config/zod/maskformer/config_2.json

python train_maskformer.py --config config/zod/maskformer/config_3.json
python test_maskformer.py --config config/zod/maskformer/config_3.json
python visualize_maskformer.py --config config/zod/maskformer/config_3.json --upload
python benchmark_maskformer.py --config config/zod/maskformer/config_3.json

python train_maskformer.py --config config/zod/maskformer/config_4.json
python test_maskformer.py --config config/zod/maskformer/config_4.json
python visualize_maskformer.py --config config/zod/maskformer/config_4.json --upload
python benchmark_maskformer.py --config config/zod/maskformer/config_4.json

python train_maskformer.py --config config/zod/maskformer/config_5.json
python test_maskformer.py --config config/zod/maskformer/config_5.json
python visualize_maskformer.py --config config/zod/maskformer/config_5.json --upload
python benchmark_maskformer.py --config config/zod/maskformer/config_5.json

python train_maskformer.py --config config/zod/maskformer/config_6.json
python test_maskformer.py --config config/zod/maskformer/config_6.json
python visualize_maskformer.py --config config/zod/maskformer/config_6.json --upload
python benchmark_maskformer.py --config config/zod/maskformer/config_6.json

python train_maskformer.py --config config/zod/maskformer/config_7.json
python test_maskformer.py --config config/zod/maskformer/config_7.json
python visualize_maskformer.py --config config/zod/maskformer/config_7.json --upload
python benchmark_maskformer.py --config config/zod/maskformer/config_7.json

python train_maskformer.py --config config/zod/maskformer/config_8.json
python test_maskformer.py --config config/zod/maskformer/config_8.json
python visualize_maskformer.py --config config/zod/maskformer/config_8.json --upload
python benchmark_maskformer.py --config config/zod/maskformer/config_8.json

python train_maskformer.py --config config/zod/maskformer/config_9.json
python test_maskformer.py --config config/zod/maskformer/config_9.json
python visualize_maskformer.py --config config/zod/maskformer/config_9.json --upload
python benchmark_maskformer.py --config config/zod/maskformer/config_9.json
