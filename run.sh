CUDA_VISIBLE_DEVICES=4,5,6,7 
accelerate launch --num_processes 3 --config_file ./deepspeed_zero3.yaml ./run_grpo.py --config ./codellama-7b-grpo.yaml
