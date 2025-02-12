import sys, os

print(f"Current working directory: {os.getcwd()}")
sys.path.insert(0, os.getcwd())

from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
import argparse

from evaluator.llama_3b_evaluator import Llama_3_2_instruct_evaluator

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', type=str, nargs='+', help='fine-tuned models merged')
    parser.add_argument('--cache_dir', type=str, help='dir for loading models and evaluation datasets')
    parser.add_argument('--save_dir', type=str, help='output saving dir')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--device', type=str, default='cuda:0')

    return parser.parse_args()

def get_evaluators(tasks, cache_dir):
    evaluators = {}
    for task in tasks:
        evaluators[task] = Llama_3_2_instruct_evaluator(task, cache_dir)
    return evaluators


def main():
    args = parse_arguments()

    base_model_name =  "meta-llama/Llama-3.2-3B-Instruct"
    
    print(f'Downloading {base_model_name}, to {args.cache_dir}')
    tokenizer = AutoTokenizer.from_pretrained(base_model_name,  cache_dir = args.cache_dir,  padding_side="left")
    # tokenizer.padding_side = "left" 
    tokenizer.pad_token = tokenizer.eos_token # to avoid error
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name,  cache_dir = args.cache_dir)

        
    print('================ Evaluate Pretrained Model ================')
    evaluators = get_evaluators(args.tasks, args.cache_dir)
    accuracy_records = {}
    for task in args.tasks:  
        if task in ['stsb']:
            accuracy = evaluators[task].evaluate_stsb(base_model, tokenizer,  batch_size=args.batch_size, device=args.device, print_output=False)        
        else:      
            accuracy = evaluators[task].evaluate(base_model, tokenizer,  batch_size=args.batch_size, device=args.device, print_output=False)
        print(f'{task}: Accuracy : {round(accuracy, 4)}')
        accuracy_records[task] = round(accuracy, 4)
    
    ## saving results
    mistral_inst_dir = os.path.join(args.save_dir, "Llama_3b_inst")
    os.makedirs(mistral_inst_dir, exist_ok=True)

    file_name = 'pretrained.json'
    file_path = os.path.join(mistral_inst_dir, file_name)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(accuracy_records, f, indent=4, ensure_ascii=False)

    print(f'results successfully saved to {file_path}')

if __name__ == "__main__":
    main()