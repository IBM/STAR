# Evaluate Pretrained model performance
python3 ./eval/llama_3b_inst/pretrained.py --save_dir './exp_results' --cache_dir '/storage/ssd3/ArthurLee/HuggingFace' --tasks 'sst2'  'mrpc' 'wic' 'cola' 'mnli' 'stsb' 'cb' 'multirc' 'rte' 'copa' 'wsc' 'qnli' 'qqp'

# Evaluate LoRA-tuned models performance
python3 ./eval/llama_3b_inst/lora.py --save_dir './exp_results' --cache_dir '/storage/ssd3/ArthurLee/HuggingFace' --tasks 'sst2'  'mrpc' 'wic' 'cola' 'mnli' 'stsb' 'cb' 'multirc' 'rte' 'copa' 'wsc' 'qnli' 'qqp'


### Merging All 13 models
# Evaluate simple avg merging performance
python3 ./runner/llama_3b_inst.py --method 'simple_avg' --save_dir './exp_results' --cache_dir '/storage/ssd3/ArthurLee/HuggingFace' --tasks 'sst2'  'mrpc' 'wic' 'cola' 'mnli' 'stsb' 'cb' 'multirc' 'rte' 'copa' 'wsc' 'qnli' 'qqp'

# Evaluate STAR merging performance
python3 ./runner/llama_3b_inst.py --method 'STAR' --eta 40 --known_rank 16 --save_dir './exp_results' --cache_dir '/storage/ssd3/ArthurLee/HuggingFace' --tasks 'sst2'  'mrpc' 'wic' 'cola' 'mnli' 'stsb' 'cb' 'multirc' 'rte' 'copa' 'wsc' 'qnli' 'qqp'

# Evaluate TIES merging performance
python3 ./runner/llama_3b_inst.py --method 'TIES' --k 20 --save_dir './exp_results' --cache_dir '/storage/ssd3/ArthurLee/HuggingFace' --tasks 'sst2'  'mrpc' 'wic' 'cola' 'mnli' 'stsb' 'cb' 'multirc' 'rte' 'copa' 'wsc' 'qnli' 'qqp'

# Evaluate TALL-masks(+TA) merging performance
python3 ./runner/llama_3b_inst.py --method 'TALL-masks' --lambda_ 0.4 --alpha 0.4 --save_dir './exp_results' --cache_dir '/storage/ssd3/ArthurLee/HuggingFace' --tasks 'sst2'  'mrpc' 'wic' 'cola' 'mnli' 'stsb' 'cb' 'multirc' 'rte' 'copa' 'wsc' 'qnli' 'qqp'

# Evaluate MetaGPT merging performance
python3 ./runner/llama_3b_inst.py --method 'MetaGPT' --save_dir './exp_results' --cache_dir '/storage/ssd3/ArthurLee/HuggingFace' --tasks 'sst2'  'mrpc' 'wic' 'cola' 'mnli' 'stsb' 'cb' 'multirc' 'rte' 'copa' 'wsc' 'qnli' 'qqp'



### Merging Random 9 models
# Evaluate simple avg merging performance
python3 ./runner/llama_3b_inst.py --method 'simple_avg' --save_dir './exp_results' --cache_dir '/storage/ssd3/ArthurLee/HuggingFace' --tasks 'sst2'  'mrpc' 'wic' 'cola' 'mnli' 'stsb' 'cb' 'multirc' 'rte'

# Evaluate STAR merging performance
python3 ./runner/llama_3b_inst.py --method 'STAR' --eta 40 --known_rank 16 --save_dir './exp_results' --cache_dir '/storage/ssd3/ArthurLee/HuggingFace' --tasks 'sst2'  'mrpc' 'wic' 'cola' 'mnli' 'stsb' 'cb' 'multirc' 'rte'

# Evaluate TIES merging performance
python3 ./runner/llama_3b_inst.py --method 'TIES' --k 20 --save_dir './exp_results' --cache_dir '/storage/ssd3/ArthurLee/HuggingFace' --tasks 'sst2'  'mrpc' 'wic' 'cola' 'mnli' 'stsb' 'cb' 'multirc' 'rte'

# Evaluate TALL-masks(+TA) merging performance
python3 ./runner/llama_3b_inst.py --method 'TALL-masks' --lambda_ 0.4 --alpha 0.4 --save_dir './exp_results' --cache_dir '/storage/ssd3/ArthurLee/HuggingFace' --tasks 'sst2'  'mrpc' 'wic' 'cola' 'mnli' 'stsb' 'cb' 'multirc' 'rte'

# Evaluate MetaGPT merging performance
python3 ./runner/llama_3b_inst.py --method 'MetaGPT' --save_dir './exp_results' --cache_dir '/storage/ssd3/ArthurLee/HuggingFace' --tasks 'sst2'  'mrpc' 'wic' 'cola' 'mnli' 'stsb' 'cb' 'multirc' 'rte'


### Merging Random 6 models
# Evaluate simple avg merging performance
python3 ./runner/llama_3b_inst.py --method 'simple_avg' --save_dir './exp_results' --cache_dir '/storage/ssd3/ArthurLee/HuggingFace' --tasks 'sst2'  'mrpc' 'wic' 'cola' 'mnli' 'stsb'

# Evaluate STAR merging performance
python3 ./runner/llama_3b_inst.py --method 'STAR' --eta 40 --known_rank 16 --save_dir './exp_results' --cache_dir '/storage/ssd3/ArthurLee/HuggingFace' --tasks 'sst2'  'mrpc' 'wic' 'cola' 'mnli' 'stsb'

# Evaluate TIES merging performance
python3 ./runner/llama_3b_inst.py --method 'TIES' --k 20 --save_dir './exp_results' --cache_dir '/storage/ssd3/ArthurLee/HuggingFace' --tasks 'sst2'  'mrpc' 'wic' 'cola' 'mnli' 'stsb'

# Evaluate TALL-masks(+TA) merging performance
python3 ./runner/llama_3b_inst.py --method 'TALL-masks' --lambda_ 0.4 --alpha 0.4 --save_dir './exp_results' --cache_dir '/storage/ssd3/ArthurLee/HuggingFace' --tasks 'sst2'  'mrpc' 'wic' 'cola' 'mnli' 'stsb'

# Evaluate MetaGPT merging performance
python3 ./runner/llama_3b_inst.py --method 'MetaGPT' --save_dir './exp_results' --cache_dir '/storage/ssd3/ArthurLee/HuggingFace' --tasks 'sst2'  'mrpc' 'wic' 'cola' 'mnli' 'stsb'


### Merging Random 6 models_2
# Evaluate simple avg merging performance
python3 ./runner/llama_3b_inst.py --method 'simple_avg' --save_dir './exp_results' --cache_dir '/storage/ssd3/ArthurLee/HuggingFace' --tasks 'sst2'  'mrpc' 'wic' 'cola' 'qnli' 'stsb'

# Evaluate STAR merging performance
python3 ./runner/llama_3b_inst.py --method 'STAR' --eta 40 --known_rank 16 --save_dir './exp_results' --cache_dir '/storage/ssd3/ArthurLee/HuggingFace' --tasks 'sst2'  'mrpc' 'wic' 'cola' 'qnli' 'stsb'

# Evaluate TIES merging performance
python3 ./runner/llama_3b_inst.py --method 'TIES' --k 20 --save_dir './exp_results' --cache_dir '/storage/ssd3/ArthurLee/HuggingFace' --tasks 'sst2'  'mrpc' 'wic' 'cola' 'qnli' 'stsb'

# Evaluate TALL-masks(+TA) merging performance
python3 ./runner/llama_3b_inst.py --method 'TALL-masks' --lambda_ 0.4 --alpha 0.4 --save_dir './exp_results' --cache_dir '/storage/ssd3/ArthurLee/HuggingFace' --tasks 'sst2'  'mrpc' 'wic' 'cola' 'qnli' 'stsb'

# Evaluate MetaGPT merging performance
python3 ./runner/llama_3b_inst.py --method 'MetaGPT' --save_dir './exp_results' --cache_dir '/storage/ssd3/ArthurLee/HuggingFace' --tasks 'sst2'  'mrpc' 'wic' 'cola' 'qnli' 'stsb'