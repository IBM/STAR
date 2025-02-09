import sys, os

print(f"Current working directory: {os.getcwd()}")
sys.path.insert(0, os.getcwd())

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json
import os
import argparse

from evaluator.flan_t5_large_evaluator import Flan_t5_Large_evaluator


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', type=str, nargs='+', help='fine-tuned models merged')
    parser.add_argument('--cache_dir', type=str, help='dir for loading models and evaluation datasets')
    parser.add_argument('--save_dir', type=str, help='output saving dir')
    parser.add_argument('--device', type=str, default='cuda:0')

    return parser.parse_args()


def get_evaluators(tasks, cache_dir):
    evaluators = {}
    for task in tasks:
        evaluators[task] = Flan_t5_Large_evaluator(task, cache_dir)
    return evaluators



def main():
    args = parse_arguments()

    base_model_name =  "google/flan-t5-large"
    
    print(f'Downloading Flan-T5-large, to {args.cache_dir}')
    tokenizer = AutoTokenizer.from_pretrained(base_model_name,  cache_dir = args.cache_dir)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name,  cache_dir = args.cache_dir)

        
        
    print('================ Evaluate Pretrained Model ================')
    evaluators = get_evaluators(args.tasks, args.cache_dir)
    accuracy_records = {}
    for task in args.tasks:        
        print(f'Evaluating on {task}')
        if task == 'stsb':
            spearman_rho = evaluators[task].evaluate_stsb(base_model, tokenizer, device=args.device, print_output=False,  batch_size = 16)        
            print(f'{task}: Spearman_rho : {round(spearman_rho, 4)}')
            accuracy_records[task] = round(spearman_rho, 4)
        else:
            accuracy = evaluators[task].evaluate(base_model, tokenizer, device=args.device, print_output=False,  batch_size = 16)
            print(f'{task}: Accuracy : {round(accuracy, 4)}')
            accuracy_records[task] = round(accuracy, 4)


    ## saving results
    flan_t5_large_dir = os.path.join(args.save_dir, "Flan_T5_large")
    os.makedirs(flan_t5_large_dir, exist_ok=True)

    file_name = 'pretrained.json'
    file_path = os.path.join(flan_t5_large_dir, file_name)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(accuracy_records, f, indent=4, ensure_ascii=False)

    print(f'results successfully saved to {file_path}')


if __name__ == "__main__":
    main()