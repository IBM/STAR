import sys, os

print(f"Current working directory: {os.getcwd()}")
sys.path.insert(0, os.getcwd())

from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
import argparse

from util import load_lora_into_base, clean_model_out
from task_vector.get_ft_models import load_llama_3b_ft_tvs
from evaluator.llama_3b_evaluator import Llama_3_2_instruct_evaluator

from merger.STAR import star_task_vectors
from merger.TIES import local_trim_weights, global_trim_weights, elect_sign, disjoint_merge
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
    parser.add_argument('--batch_size', type=int, default=8, help='batch size used when batch inference to evaluate model performance')


    ## STAR specific hyperparameter
    parser.add_argument('--eta', type=float, default=40, help='truncate eta percentage of nuclear norm')
    parser.add_argument('--known_rank', type=int, help='specify the rank of LoRA here, for faster SVD computation')

    ## TIES specific hyperparameter
    parser.add_argument('--k', type=float, default=20, help='trimming top-k percentage of parameters')
    parser.add_argument('--trim', type=str, default='global', choices=['local', 'global'], help='trim matrix by matrix or globally')

    ## Tall-mask specific hyperparameters
    parser.add_argument('--lambda_', type=float, default=0.4, help='') ## Freeze this, as tuning this requirs data
    parser.add_argument('--alpha', type=float, default=0.4, help='the task aritmetic related scaling') ## recommended value from [Editing Models with Task Arithmetic]
    return parser.parse_args()


def get_evaluators(tasks, cache_dir):
    evaluators = {}
    for task in tasks:
        evaluators[task] = Llama_3_2_instruct_evaluator(task, cache_dir)
    return evaluators

def evaluate_model(model, tokenizer, evaluators, tasks, batch_size, device):
    accuracy_records = {}
    for task in tasks:
        # print(f'Evaluating on {task}')
        if task in ['stsb']:
            accuracy = evaluators[task].evaluate_stsb(model, tokenizer,  batch_size=batch_size, device=device, print_output=False)        
        else:      
            accuracy = evaluators[task].evaluate(model, tokenizer,  batch_size=batch_size, device=device, print_output=False)
        print(f'{task}: Accuracy : {round(accuracy, 4)}')
        accuracy_records[task] = round(accuracy, 4)
    return accuracy_records

def main():
    args = parse_arguments()

    base_model_name =  "meta-llama/Llama-3.2-3B-Instruct"
    
    print(f'Downloading {base_model_name}, to {args.cache_dir}')
    tokenizer = AutoTokenizer.from_pretrained(base_model_name,  cache_dir = args.cache_dir)
    tokenizer.padding_side = "left" 
    tokenizer.pad_token = tokenizer.eos_token # to avoid an error
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name,  cache_dir = args.cache_dir)


    print(f'Perform merging on {args.tasks}')
    print(f'Total: {len(args.tasks)} models')
    
    print('================ Get the task vectors ================')
    tv_dicts = load_llama_3b_ft_tvs(base_model, args.tasks, args.cache_dir)
    sorted_task_str = "_".join(sorted(args.tasks, key=lambda x: x[0]))
    
    if args.method == 'STAR':
        print(f'================ STAR merging using eta = {args.eta} ================')
        file_name = f'{args.method}_eta{args.eta}.json'
        if args.known_rank is not None:
            print('activate low_rank SVD')
        else:
            print('full SVD is used, please specify args.known_rank for faster low_rank SVD')
            
        star_tv_dicts = star_task_vectors(tv_dicts, args.eta, known_rank=args.known_rank) 
        star_tv_list = list(star_tv_dicts.values())
        
        ## Simple Averaging
        alphas = [1/len(star_tv_list)] * len(star_tv_list) 
        merged_tv_dict = weighted_merge(star_tv_list, alphas)
    elif args.method == 'TIES':
        print(f'================ TIES merging using k = {args.k} ================')
        file_name = f'{args.method}_k{args.k}.json'
        # Trim
        if args.trim == 'local':
            trimmed_tv_dicts = local_trim_weights(tv_dicts, args.k, args.device)
        elif args.trim == 'global':
            trimmed_tv_dicts = global_trim_weights(tv_dicts, args.k, args.device)
        # Elect Sign
        sign_reference_dict = elect_sign(trimmed_tv_dicts)
        # Disjoint Merge
        merged_tv_dict = disjoint_merge(trimmed_tv_dicts, sign_reference_dict)
    elif args.method == 'TALL-masks':
        file_name = f'{args.method}_lambda{args.lambda_}_alpha_{args.alpha}.json'
        print(f'================ TALL-masks merging using lambda_ = {args.lambda_}, alpha = {args.alpha} ================')
        tv_list = list(tv_dicts.values())
        merged_tv_dict = get_merged_tv_tall(tv_list, args.lambda_, args.alpha)
    elif args.method == 'MetaGPT':
        print(f'================ MetaGPT merging ================')
        file_name = f'{args.method}.json'
        tv_list = list(tv_dicts.values())
        merged_tv_dict = metagpt_get_merged_tv(tv_list)
    elif args.method == 'simple_avg':
        print(f'================ simple_avg merging ================')
        file_name = f'{args.method}.json'
        tv_list = list(tv_dicts.values())
        alphas = [1/len(tv_list)] * len(tv_list) 
        merged_tv_dict = weighted_merge(tv_list, alphas)
    
        
    print('================ Loading merged weigths into Pre-trained Model ================')
    merged_model = load_lora_into_base(base_model, merged_tv_dict)    
    
    
    print('================ Evaluate Merged Model ================')
    evaluators = get_evaluators(args.tasks, args.cache_dir)
    accuracy_records = evaluate_model(merged_model, tokenizer, evaluators, args.tasks, args.batch_size, args.device)
    
    ## saving results
    llama_3b_dir = os.path.join(args.save_dir, f"Llama_3b_inst/{sorted_task_str}")
    os.makedirs(llama_3b_dir, exist_ok=True)


    file_path = os.path.join(llama_3b_dir, file_name)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(accuracy_records, f, indent=4, ensure_ascii=False)

    print(f'results successfully saved to {file_path}')



    clean_model_out(merged_model)

if __name__ == "__main__":
    main()