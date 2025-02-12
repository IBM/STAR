import sys, os

print(f"Current working directory: {os.getcwd()}")
sys.path.insert(0, os.getcwd())


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from task_vector.get_ft_models import load_flan_t5_large_ft_tvs
from evaluator.flan_t5_large_evaluator import Flan_t5_Large_evaluator
from util import load_lora_into_base

def get_evaluators(tasks, cache_dir):
    evaluators = {}
    for task in tasks:
        evaluators[task] = Flan_t5_Large_evaluator(task, cache_dir)
    return evaluators

def evaluate_model(model, tokenizer, evaluators, tasks, device):
    accuracy_records = {}
    for task in tasks:
        print(f'Evaluating on {task}')
        if task == 'stsb':
            spearman_rho = evaluators[task].evaluate_stsb(model, tokenizer, device=device, print_output=False,  batch_size = 16)        
            print(f'{task}: Spearman_rho : {round(spearman_rho, 4)}')
            accuracy_records[task] = round(spearman_rho, 4)
        else:
            accuracy = evaluators[task].evaluate(model, tokenizer, device=device, print_output=False,  batch_size = 16)
            print(f'{task}: Accuracy : {round(accuracy, 4)}')
            accuracy_records[task] = round(accuracy, 4)
    return accuracy_records



cache_dir = '/storage/ssd3/ArthurLee/HuggingFace'
device = 'cuda:0'
tasks = ['mrpc']


base_model_name =  "google/flan-t5-large"

print(f'Downloading Flan-T5-large, to {cache_dir}')
tokenizer = AutoTokenizer.from_pretrained(base_model_name,  cache_dir = cache_dir)
base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name,  cache_dir = cache_dir)


tv_dicts = load_flan_t5_large_ft_tvs(tasks, cache_dir)


# print(base_model.state_dict().keys())

# print(tv_dicts['mrpc'].keys())

print(tv_dicts['mrpc']['decoder.block.15.layer.1.EncDecAttention.q.weight'])

merged_model = load_lora_into_base(base_model, tv_dicts['mrpc'])    
    


print('================ Evaluate Merged Model ================')
evaluators = get_evaluators(tasks, cache_dir)
accuracy_records = evaluate_model(merged_model, tokenizer, evaluators, tasks, device)
