# STAR: Spectral Truncation and Rescale for Model Merging ([NAACL 2025](https://2025.naacl.org/))

![Flow Diagram](./image/flow.jpg)

## Setup
```bash
# 1. Clone this repo
git clone https://github.com/IBM/STAR.git
cd STAR

# 2. Create a virtual environment and activate it
conda create --name STAR-env python=3.9
conda activate STAR-env

# 3. Install dependencies
pip install -r requirements.txt
```
Please also find a local directory (i.e. `<your_cahce_dir>`) with enough space for storing task vectors and datasets.


## Perform Model Merging
### Flan-T5-large
#### Supported 13 fine-tuned models:
`mnli`, `mrpc`, `qnli`, `qnli`, `qqp`, `rte`, `sst2`, `stsb`, `finance`, `imdb`, `agnews`, `hella`, `boolq`, `piqa`

────────────────────────────────────
#### Evaluate Pretrained Model Performance (lower bound):
```bash
# Evaluate Pretrained model performance all downstream tasks
python3 ./eval/flan_t5_large/pretrained.py --save_dir './exp_results' --cache_dir '<your_cahce_dir>' --tasks 'mnli' 'mrpc' 'qnli' 'qqp' 'rte' 'sst2' 'stsb' 'finance' 'imdb' 'agnews' 'hella' 'boolq' 'piqa'
```
with the results at 
```bash
'./exp_results/Flan_T5_large/pretrained.json'
```
*The first run will take more time for loading models and datasets, but it will be faster thereafter.*
────────────────────────────────────
#### Evaluate LoRA Fine-tuned models Performance (upper bound):
```bash
# Evaluate each lora model performance on corresponding downstream task
python3 ./eval/flan_t5_large/lora.py --save_dir './exp_results' --cache_dir '<your_cahce_dir>' --tasks 'mnli' 'mrpc' 'qnli' 'qqp' 'rte' 'sst2' 'stsb' 'finance' 'imdb' 'agnews' 'hella' 'boolq' 'piqa'
```
with the results at 
```bash
'./exp_results/Flan_T5_large/lora.json'
```

---
#### Model merging using different methods (example on mergeing random 8):
```bash
# STAR
python3 ./runner/flan_t5_large.py --method 'STAR' --eta 40 --save_dir './exp_results' --cache_dir '<your_cahce_dir>' --tasks 'mnli' 'qnli' 'qqp' 'stsb' 'finance' 'imdb' 'boolq' 'piqa'

# simple_avg
python3 ./runner/flan_t5_large.py --method 'simple_avg' --save_dir './exp_results' --cache_dir '<your_cahce_dir>' --tasks 'mnli' 'qnli' 'qqp' 'stsb' 'finance' 'imdb' 'boolq' 'piqa'

# TIES
python3 ./runner/flan_t5_large.py --method 'TIES' --k 20 --save_dir './exp_results' --cache_dir '<your_cahce_dir>' --tasks 'mnli' 'qnli' 'qqp' 'stsb' 'finance' 'imdb' 'boolq' 'piqa'

# Tall-masks
python3 ./runner/flan_t5_large.py --method 'TALL-masks' --lambda_ 0.4 --alpha 0.3 --save_dir './exp_results' --cache_dir '<your_cahce_dir>' --tasks 'mnli' 'qnli' 'qqp' 'stsb' 'finance' 'imdb' 'boolq' 'piqa'

# MetaGPT
python3 ./runner/flan_t5_large.py --method 'MetaGPT' --save_dir './exp_results' --cache_dir '<your_cahce_dir>' --tasks 'mnli' 'qnli' 'qqp' 'stsb' 'finance' 'imdb' 'boolq' 'piqa'
```
Then, the merged model performance on each dataset using different methods can be viewed at the following json files:
```bash
'./exp_results/Flan_T5_large/STAR_eta40.0_boolq_finance_imdb_mnli_piqa_qnli_qqp_stsb.json'
'./exp_results/Flan_T5_large/simple_avg_boolq_finance_imdb_mnli_piqa_qnli_qqp_stsb.json'
'./exp_results/Flan_T5_large/TIES_k20.0_boolq_finance_imdb_mnli_piqa_qnli_qqp_stsb.json'
'./exp_results/Flan_T5_large/TALL-masks_lambda0.4_alpha_0.4_boolq_finance_imdb_mnli_piqa_qnli_qqp_stsb.json'
'./exp_results/Flan_T5_large/MetaGPT_agnews_boolq_finance_hella_imdb_mnli_mrpc_piqa_qnli_qqp_rte_sst2_stsb.json'
```
---
#### Example Merging Results (on random sampled 10)
|                | mnli  | mrpc  | qnli  | qqp   | rte   | sst2  | finance | imdb  | hella | boolq | Normalized Avg  |
|:--------------:|:-----:|:-----:|:-----:|:-----:|:-------:|:-----:|:------:|:-----:|:-----:|:-----:|:--------------:|
| Pretrained      | 80.83 | 71.45 | 91.25 | 87.85 | 85.50 | 94.15 | 55.51   | 76.80 | 74.38 | 82.50 | 89.62   |
| LoRA      | 88.50 | 83.56 | 94.55 | 87.80 | 91.98 | 94.95 | 96.48   | 95.95 | 76.88 | 85.62 | 100.00  |
| **Simple Avg**| 80.83 | 76.82 | 92.90 | 86.20 | 87.02 | 94.95 | 67.40   |  79.70 | 76.25 | 83.75|92.42 |
| **TIES**    | 63.58 | 77.51 | 92.85 | 86.95 | 86.26 | 94.15 | 76.21   | 95.90 | 76.88 | 82.50 | 93.00   |
| **TALL-masks**| 81.83 | 82.96 | 93.50 | 83.00 | 89.31 | 95.30 | 88.11  | 96.00 | 75.00| 82.50 | 96.79   |
| **MetaGPT**   |   45.92 | 70.07 | 92.15 | 87.15 | 86.26 | 94.72 | 56.83   |  76.85 | 75.62 | 73.75|  84.95   |
| **STAR**      |  81.33 | 79.76 | 93.65 | 85.05 | 88.55| 94.95 | 75.33  | 95.60 |  75.62|82.50| 95.20   |


#### Example Merging Results (on all 13)
|                | mnli  | mrpc  | qnli  | qqp   | rte   | sst2  | stsb  | finance | imdb  | agnews | hella | boolq | piqa  | Normalized Avg  |
|:--------------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-------:|:-----:|:------:|:-----:|:-----:|:-----:|:--------------:|
| Pretrained | 80.83 | 71.45 | 91.25 | 87.85 | 85.50 | 94.15 | 87.19 | 55.51   | 76.80 | 88.00  | 74.38 | 82.50  | 77.47 | 91.26   |
| LoRA       | 88.50 | 83.56 | 94.55 | 87.80 | 91.98 | 94.95 | 90.85 | 96.48   | 95.95 | 91.00  | 76.88 | 85.62  | 79.45 | 100.00   |
| **Simple Avg** | 80.08 | 76.38 | 92.65 | 86.30 | 87.02 | 94.95 | 87.28 | 62.11   | 77.05 | 89.50  | 76.25 | 83.75  | 77.75 | <u>92.83</u>   |
| **TIES**       | 55.00 | 77.08 | 92.90 | 86.80 | 85.88 | 94.50 | 85.27 | 55.95   | 95.85 | 91.00  | 78.12 | 80.00  | 77.09 | 91.40   |
| **TALL-masks** | 51.75 | 82.18 | 93.30 | 82.75 | 89.69 | 94.95 | 83.20 | 70.04   | 96.00 | 91.00  | 71.25 | 67.50  | 76.10 | 90.68   |
| **MetaGPT**    | 49.50 | 72.58 | 92.50 | 86.85 | 87.02 | 94.50 | 84.56 | 52.86   | 76.95 | 91.00  | 75.62 | 78.75  | 76.87 | 88.37   |
| **STAR**       | 80.08 | 78.63 | 93.20 | 85.30 | 88.17 | 94.95 | 87.64 | 66.96   | 88.30 | 91.00  | 76.88 | 83.13  | 77.86 | **94.55** |

*Note that the exact number might slightly vary from the paper (shfit upward in this case). This is perfectly normal since testing data was randomly sampled to avoid computation overhead.*  
*Feel free to adjust the number of samples for each dataset according to your needs [here](https://github.com/IBM/STAR/blob/cb1be15ce9c4428f8adeb6f605348436fa481a84/evaluator/flan_t5_large_evaluator.py#L34), and an identical trend could still be observed.*





