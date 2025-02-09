import sys, os

print(f"Current working directory: {os.getcwd()}")
sys.path.insert(0, os.getcwd())

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json
import os
import argparse

from task_vector.util import load_lora_into_base, clean_model_out
from task_vector.get_ft_models import load_flan_t5_large_ft_tvs
from evaluator.flan_t5_large_evaluator import Flan_t5_Large_evaluator

from merger.STAR import star_task_vectors
from merger.TIES import global_trim_weights, elect_sign, disjoint_merge
from merger.tall_mask import get_merged_tv_tall
from merger.metagpt import metagpt_get_merged_tv
from merger.simple_avg import weighted_merge



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, choices=['STAR', 'TIES', 'TALL-masks', 'MetaGPT', 'simple_avg'], help='model merging method')
    parser.add_argument('--tasks', type=str, nargs='+', help='fine-tuned models merged')
    parser.add_argument('--cache_dir', type=str, help='dir for loading models and evaluation datasets')
    parser.add_argument('--save_dir', type=str, help='output saving dir')
    parser.add_argument('--device', type=str, default='cuda:0')

    ## STAR specific hyperparameter
    parser.add_argument('--eta', type=float, default=40, help='truncate eta percentage of nuclear norm')

    ## TIES specific hyperparameter
    parser.add_argument('--k', type=float, default=20, help='trimming top-k percentage of parameters')

    ## Tall-mask specific hyperparameters
    parser.add_argument('--lambda_', type=float, default=0.4, help='') ## Freeze this, as tuning this requirs data
    parser.add_argument('--alpha', type=float, default=0.3, help='the task aritmetic related scaling') ## recommended value from [Editing Models with Task Arithmetic]
    return parser.parse_args()


def get_evaluators(tasks, cache_dir):
    evaluators = {}
    for task in tasks:
        evaluators[task] = Flan_t5_Large_evaluator(task, cache_dir)
    return evaluators

def evaluate_model(model, tokenizer, evaluators, tasks, device):
    accuracy_records = {}
    for task in tasks:
        # print(f'Evaluating on {task}')
        if task == 'stsb':
            spearman_rho = evaluators[task].evaluate_stsb(model, tokenizer, device=device, print_output=False,  batch_size = 16)        
            print(f'{task}: Spearman_rho : {round(spearman_rho, 4)}')
            accuracy_records[task] = round(spearman_rho, 4)
        else:
            accuracy = evaluators[task].evaluate(model, tokenizer, device=device, print_output=False,  batch_size = 16)
            print(f'{task}: Accuracy : {round(accuracy, 4)}')
            accuracy_records[task] = round(accuracy, 4)
    return accuracy_records

def main():
    args = parse_arguments()

    base_model_name =  "google/flan-t5-large"
    
    print(f'Downloading Flan-T5-large, to {args.cache_dir}')
    tokenizer = AutoTokenizer.from_pretrained(base_model_name,  cache_dir = args.cache_dir)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name,  cache_dir = args.cache_dir)

    
    print('================ Get the task vectors ================')
    tv_dicts = load_flan_t5_large_ft_tvs(base_model, args.tasks, args.cache_dir)
    sorted_task_str = "_".join(sorted(args.tasks, key=lambda x: x[0]))
    
    if args.method == 'STAR':
        print(f'================ STAR merging using eta = {args.eta} ================')
        file_name = f'{args.method}_eta{args.eta}_{sorted_task_str}.json'

        star_tv_dicts = star_task_vectors(tv_dicts, args.eta)
        star_tv_list = list(star_tv_dicts.values())
        
        ## Simple Averaging
        alphas = [1/len(star_tv_list)] * len(star_tv_list) 
        merged_tv_dict = weighted_merge(star_tv_list, alphas)

    elif args.method == 'TIES':
        print(f'================ TIES merging using k = {args.k} ================')
        file_name = f'{args.method}_k{args.k}_{sorted_task_str}.json'
        # Trim
        trimmed_tv_dicts = global_trim_weights(tv_dicts, args.k)
        #  {task: global_trim_weights(tv, args.k) for task, tv in tv_dicts.items()}
        # Elect Sign
        sign_reference_dict = elect_sign(trimmed_tv_dicts)
        # Disjoint Merge
        merged_tv_dict = disjoint_merge(trimmed_tv_dicts, sign_reference_dict)
    elif args.method == 'TALL-masks':
        file_name = f'{args.method}_lambda{args.lambda_}_alpha_{args.alpha}_{sorted_task_str}.json'
        print(f'================ TALL-masks merging using lambda_ = {args.lambda_}, alpha = {args.alpha} ================')
        tv_list = list(tv_dicts.values())
        merged_tv_dict = get_merged_tv_tall(tv_list, args.lambda_, args.alpha)
    elif args.method == 'MetaGPT':
        print(f'================ MetaGPT merging ================')
        file_name = f'{args.method}_{sorted_task_str}.json'
        tv_list = list(tv_dicts.values())
        merged_tv_dict = metagpt_get_merged_tv(tv_list)
    elif args.method == 'simple_avg':
        print(f'================ simple_avg merging ================')
        file_name = f'{args.method}_{sorted_task_str}.json'
        tv_list = list(tv_dicts.values())
        alphas = [1/len(tv_list)] * len(tv_list) 
        merged_tv_dict = weighted_merge(tv_list, alphas)
    
        
    print('================ Loading merged weigths into Pre-trained Model ================')
    merged_model = load_lora_into_base(base_model, merged_tv_dict)    
    
    
    print('================ Evaluate Merged Model ================')
    evaluators = get_evaluators(args.tasks, args.cache_dir)
    accuracy_records = evaluate_model(merged_model, tokenizer, evaluators, args.tasks, args.device)
    
    ## saving results
    flan_t5_large_dir = os.path.join(args.save_dir, "Flan_T5_large")
    os.makedirs(flan_t5_large_dir, exist_ok=True)


    file_path = os.path.join(flan_t5_large_dir, file_name)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(accuracy_records, f, indent=4, ensure_ascii=False)

    print(f'results successfully saved to {file_path}')



    clean_model_out(merged_model)

if __name__ == "__main__":
    main()