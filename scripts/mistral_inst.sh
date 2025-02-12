# Evaluate Pretrained model performance
python3 ./eval/mistral_inst/pretrained.py --batch_size 8 --save_dir './exp_results' --cache_dir '/storage/ssd3/ArthurLee/HuggingFace' --tasks 'ethos' 'wino' 'stereo' 'causal' 'answerable' 'qasc' 'dream' 'ncbi' 'owant' 'amazon' 'msr' 'gap' 'snli' 'argue' 'disco' 'math' 'casino' 'story' 'pubmed' 'sst2'



# Evaluate LoRA-tuned models performance
python3 ./eval/mistral_inst/lora.py --batch_size 8 --save_dir './exp_results' --cache_dir '/storage/ssd3/ArthurLee/HuggingFace' --tasks 'ethos' 'wino' 'stereo' 'causal' 'answerable' 'dream' 'ncbi' 'owant' 'amazon' 'msr' 'gap' 'snli' 'argue' 'disco' 'math' 'casino' 'story' 'pubmed' 'sst2'
 
# 'answerable' 更改過版本，performance掉了
# 'qasc' 好像被Lots of LoRAs下架了


### Merging All 18 models
# Evaluate simple avg merging performance
python3 ./runner/mistral_inst.py --method 'simple_avg' --save_dir './exp_results' --cache_dir '/storage/ssd3/ArthurLee/HuggingFace' --tasks 'ethos' 'wino' 'stereo' 'causal' 'dream' 'ncbi' 'owant' 'amazon' 'msr' 'gap' 'snli' 'argue' 'disco' 'math' 'casino' 'story' 'pubmed' 'sst2'

# Evaluate STAR merging performance
python3 ./runner/mistral_inst.py --method 'STAR' --eta 40 --save_dir './exp_results' --cache_dir '/storage/ssd3/ArthurLee/HuggingFace' --tasks 'ethos' 'wino' 'stereo' 'causal' 'dream' 'ncbi' 'owant' 'amazon' 'msr' 'gap' 'snli' 'argue' 'disco' 'math' 'casino' 'story' 'pubmed' 'sst2'

# Evaluate TIES merging performance
python3 ./runner/mistral_inst.py --method 'TIES' --k 20 --save_dir './exp_results' --cache_dir '/storage/ssd3/ArthurLee/HuggingFace' --tasks 'ethos' 'wino' 'stereo' 'causal' 'dream' 'ncbi' 'owant' 'amazon' 'msr' 'gap' 'snli' 'argue' 'disco' 'math' 'casino' 'story' 'pubmed' 'sst2'

# Evaluate TALL-masks(+TA) merging performance
python3 ./runner/mistral_inst.py --method 'TALL-masks' --lambda_ 0.4 --alpha 0.4 --save_dir './exp_results' --cache_dir '/storage/ssd3/ArthurLee/HuggingFace' --tasks 'ethos' 'wino' 'stereo' 'causal' 'dream' 'ncbi' 'owant' 'amazon' 'msr' 'gap' 'snli' 'argue' 'disco' 'math' 'casino' 'story' 'pubmed' 'sst2'

# Evaluate MetaGPT merging performance
python3 ./runner/mistral_inst.py --method 'MetaGPT' --save_dir './exp_results' --cache_dir '/storage/ssd3/ArthurLee/HuggingFace' --tasks 'ethos' 'wino' 'stereo' 'causal' 'dream' 'ncbi' 'owant' 'amazon' 'msr' 'gap' 'snli' 'argue' 'disco' 'math' 'casino' 'story' 'pubmed' 'sst2'



### Merging sampled 12 models
# Evaluate simple avg merging performance
python3 ./runner/mistral_inst.py --method 'simple_avg' --save_dir './exp_results' --cache_dir '/storage/ssd3/ArthurLee/HuggingFace' --tasks 'ethos' 'causal' 'ncbi' 'owant' 'gap' 'snli' 'disco' 'math' 'casino' 'story' 'pubmed' 'sst2'

# Evaluate STAR merging performance
python3 ./runner/mistral_inst.py --method 'STAR' --eta 40 --save_dir './exp_results' --cache_dir '/storage/ssd3/ArthurLee/HuggingFace' --tasks 'ethos' 'causal' 'ncbi' 'owant' 'gap' 'snli' 'disco' 'math' 'casino' 'story' 'pubmed' 'sst2'

# Evaluate TIES merging performance
python3 ./runner/mistral_inst.py --method 'TIES' --k 20 --save_dir './exp_results' --cache_dir '/storage/ssd3/ArthurLee/HuggingFace' --tasks 'ethos' 'causal' 'ncbi' 'owant' 'gap' 'snli' 'disco' 'math' 'casino' 'story' 'pubmed' 'sst2'

# Evaluate TALL-masks(+TA) merging performance
python3 ./runner/mistral_inst.py --method 'TALL-masks' --lambda_ 0.4 --alpha 0.4 --save_dir './exp_results' --cache_dir '/storage/ssd3/ArthurLee/HuggingFace' --tasks 'ethos' 'causal' 'ncbi' 'owant' 'gap' 'snli' 'disco' 'math' 'casino' 'story' 'pubmed' 'sst2'

# Evaluate MetaGPT merging performance
python3 ./runner/mistral_inst.py --method 'MetaGPT' --save_dir './exp_results' --cache_dir '/storage/ssd3/ArthurLee/HuggingFace' --tasks 'ethos' 'causal' 'ncbi' 'owant' 'gap' 'snli' 'disco' 'math' 'casino' 'story' 'pubmed' 'sst2'