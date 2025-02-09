
# Evaluate Pretrained model performance
python3 ./eval/flan_t5_large/pretrained.py --save_dir './exp_results' --cache_dir '/storage/ssd3/ArthurLee/HuggingFace' --tasks 'mnli' 'mrpc' 'qnli' 'qqp' 'rte' 'sst2' 'stsb' 'finance' 'imdb' 'agnews' 'hella' 'boolq' 'piqa'


### Merging All 13 models
# Evaluate LoRA-tuned models performance
python3 ./eval/flan_t5_large/lora.py --save_dir './exp_results' --cache_dir '/storage/ssd3/ArthurLee/HuggingFace' --tasks 'mnli' 'mrpc' 'qnli' 'qqp' 'rte' 'sst2' 'stsb' 'finance' 'imdb' 'agnews' 'hella' 'boolq' 'piqa'

# Evaluate simple avg merging performance
python3 ./runner/flan_t5_large.py --method 'simple_avg' --save_dir './exp_results' --cache_dir '/storage/ssd3/ArthurLee/HuggingFace' --tasks 'mnli' 'mrpc' 'qnli' 'qqp' 'rte' 'sst2' 'stsb' 'finance' 'imdb' 'agnews' 'hella' 'boolq' 'piqa'

# Evaluate STAR merging performance
python3 ./runner/flan_t5_large.py --method 'STAR' --eta 40 --save_dir './exp_results' --cache_dir '/storage/ssd3/ArthurLee/HuggingFace' --tasks 'mnli' 'mrpc' 'qnli' 'qqp' 'rte' 'sst2' 'stsb' 'finance' 'imdb' 'agnews' 'hella' 'boolq' 'piqa'

# Evaluate TIES merging performance
python3 ./runner/flan_t5_large.py --method 'TIES' --k 20 --save_dir './exp_results' --cache_dir '/storage/ssd3/ArthurLee/HuggingFace' --tasks 'mnli' 'mrpc' 'qnli' 'qqp' 'rte' 'sst2' 'stsb' 'finance' 'imdb' 'agnews' 'hella' 'boolq' 'piqa'

# Evaluate TALL-masks(+TA) merging performance
python3 ./runner/flan_t5_large.py --method 'TALL-masks' --lambda_ 0.4 --alpha 0.4 --save_dir './exp_results' --cache_dir '/storage/ssd3/ArthurLee/HuggingFace' --tasks 'mnli' 'mrpc' 'qnli' 'qqp' 'rte' 'sst2' 'stsb' 'finance' 'imdb' 'agnews' 'hella' 'boolq' 'piqa'

# Evaluate MetaGPT merging performance
python3 ./runner/flan_t5_large.py --method 'MetaGPT' --save_dir './exp_results' --cache_dir '/storage/ssd3/ArthurLee/HuggingFace' --tasks 'mnli' 'mrpc' 'qnli' 'qqp' 'rte' 'sst2' 'stsb' 'finance' 'imdb' 'agnews' 'hella' 'boolq' 'piqa'

### Merging sampled 8 models
# Evaluate simple avg merging performance
python3 ./runner/flan_t5_large.py --method 'simple_avg' --save_dir './exp_results' --cache_dir '/storage/ssd3/ArthurLee/HuggingFace' --tasks 'mnli' 'qnli' 'qqp'  'stsb' 'finance' 'imdb' 'boolq' 'piqa'

# Evaluate STAR merging performance
python3 ./runner/flan_t5_large.py --method 'STAR' --eta 40 --save_dir './exp_results' --cache_dir '/storage/ssd3/ArthurLee/HuggingFace' --tasks 'mnli' 'qnli' 'qqp'  'stsb' 'finance' 'imdb' 'boolq' 'piqa'

# Evaluate TIES merging performance
python3 ./runner/flan_t5_large.py --method 'TIES' --k 20 --save_dir './exp_results' --cache_dir '/storage/ssd3/ArthurLee/HuggingFace' --tasks 'mnli' 'qnli' 'qqp'  'stsb' 'finance' 'imdb' 'boolq' 'piqa'

# Evaluate TALL-masks(+TA) merging performance
python3 ./runner/flan_t5_large.py --method 'TALL-masks' --lambda_ 0.4 --alpha 0.4 --save_dir './exp_results' --cache_dir '/storage/ssd3/ArthurLee/HuggingFace' --tasks 'mnli' 'qnli' 'qqp'  'stsb' 'finance' 'imdb' 'boolq' 'piqa'

# Evaluate MetaGPT merging performance
python3 ./runner/flan_t5_large.py --method 'MetaGPT' --save_dir './exp_results' --cache_dir '/storage/ssd3/ArthurLee/HuggingFace' --tasks 'mnli' 'qnli' 'qqp'  'stsb' 'finance' 'imdb' 'boolq' 'piqa'


### Merging sampled 4 models
# Evaluate simple avg merging performance
python3 ./runner/flan_t5_large.py --method 'simple_avg' --save_dir './exp_results' --cache_dir '/storage/ssd3/ArthurLee/HuggingFace' --tasks 'mnli' 'qnli' 'finance' 'boolq'

# Evaluate STAR merging performance
python3 ./runner/flan_t5_large.py --method 'STAR' --eta 40 --save_dir './exp_results' --cache_dir '/storage/ssd3/ArthurLee/HuggingFace' --tasks 'mnli' 'qnli' 'finance' 'boolq'

# Evaluate TIES merging performance
python3 ./runner/flan_t5_large.py --method 'TIES' --k 20 --save_dir './exp_results' --cache_dir '/storage/ssd3/ArthurLee/HuggingFace' --tasks 'mnli' 'qnli' 'finance' 'boolq'

# Evaluate TALL-masks(+TA) merging performance
python3 ./runner/flan_t5_large.py --method 'TALL-masks' --lambda_ 0.4 --alpha 0.4 --save_dir './exp_results' --cache_dir '/storage/ssd3/ArthurLee/HuggingFace' --tasks 'mnli' 'qnli' 'finance' 'boolq'

# Evaluate MetaGPT merging performance
python3 ./runner/flan_t5_large.py --method 'MetaGPT' --save_dir './exp_results' --cache_dir '/storage/ssd3/ArthurLee/HuggingFace' --tasks 'mnli' 'qnli' 'finance' 'boolq'