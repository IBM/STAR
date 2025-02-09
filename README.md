# STAR: Spectral Truncation and Rescale for Model Merging ([NAACL 2025](https://2025.naacl.org/))

![Flow Diagram](./image/flow.jpg)

## Setup
```bash
conda create --name STAR-env python=3.9
conda activate STAR-env
pip install -r requirements.txt
```

## Perform Model Merging

### Flan-T5-base

### Flan-T5-large
```bash
# STAR

python3 ./runner/flan_t5_large.py --method 'STAR' --eta 40 --save_dir './exp_results' --cache_dir '/storage/ssd3/ArthurLee/HuggingFace' --tasks 'mnli' 'mrpc' 'qnli' 'rte' 'sst2' 'stsb' 'cola'

python3 ./runner/flan_t5_large.py --method 'STAR' --eta 40 --save_dir './exp_results' --cache_dir '{your directory for saving models and datasets}' --tasks 'mnli' 'mrpc' 'qnli' 'rte' 'sst2' 'stsb' 'cola'

# TIES
python3 ./runner/flan_t5_large.py --method 'TIES' --k 20 --save_dir './exp_results' --cache_dir '{your directory for saving models and datasets}' --tasks 'mnli' 'mrpc' 'qnli' 'rte' 'sst2' 'stsb' 'cola'

# Tall-masks
python3 ./runner/flan_t5_large.py --method 'TALL-masks' --lambda_ 0.4 --alpha 0.3 --save_dir './exp_results' --cache_dir '{your directory for saving models and datasets}' --tasks 'mnli' 'mrpc' 'qnli' 'rte' 'sst2' 'stsb' 'cola'

# MetaGPT
python3 ./runner/flan_t5_large.py --method 'MetaGPT' --save_dir './exp_results' --cache_dir '{your directory for saving models and datasets}' --tasks 'mnli' 'mrpc' 'qnli' 'rte' 'sst2' 'stsb' 'cola'

```
**Merging Results**


```bash
python3 ./runner/flan_t5_large.py --method 'STAR' --eta 40 --device 'cuda:0' --save_dir './exp_results' --cache_dir '/storage/ssd3/ArthurLee/HuggingFace' --tasks 'mnli' 'mrpc' 'qnli' 'rte' 'sst2' 'stsb' 'cola'
```

**Support fine-tuned models:**  
`mnli`, `mrpc`, `qnli`, `qnli`, `qqp`, `rte`, `sst2`, `stsb`, `finance`, `imdb`, `agnews`, `hella`, `boolq`, `piqa`

### Mistral Instruct

### Llama-3.2-1B-Instruct
