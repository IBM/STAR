import gc
import torch
import copy
from tqdm import tqdm
import random
from collections import Counter
from datasets import Dataset

def load_lora_into_base(base_model, lora_parameter_base_form_dict):
    base_model_for_lora = copy.deepcopy(base_model)
    base_state_dict = base_model_for_lora.cpu().state_dict()
    
    for base_name, lora_parameter in tqdm(lora_parameter_base_form_dict.items(), desc="Loading task vectors to base model"):
        if base_name in base_state_dict.keys():
            # print(f'loading lora weight {base_name} into pre-trained model')
            base_state_dict[base_name] += lora_parameter
    base_model_for_lora.load_state_dict(base_state_dict)
    base_model_for_lora.eval()
    
    return base_model_for_lora

def clean_model_out(model_variable):
    model_variable.to('cpu')
    model_variable = None
    gc.collect()
    torch.cuda.empty_cache()

    return

def random_balanced_sample(dataset, label_column, num_samples_per_label, remove_columns=None):  
    label_counts = Counter(dataset[label_column])
    min_count = min(label_counts.values())
    
    num_samples_per_label = min(num_samples_per_label, min_count)
    
    sampled_data = []
    random.seed(42) # set random seed for sampling testing data
    for label in label_counts.keys():
        label_samples = [example for example in dataset if example[label_column] == label]
        sampled_data.extend(random.sample(label_samples, num_samples_per_label))

    sampled_dataset = Dataset.from_dict({key: [sample[key] for sample in sampled_data] for key in dataset.column_names})

    if remove_columns:
        sampled_dataset = sampled_dataset.remove_columns(remove_columns)
    
    return sampled_dataset

def random_sample(dataset, num_samples, remove_columns=None):
    random.seed(42)  # set random seed for sampling testing data
    sampled_data = random.sample(list(dataset), min(num_samples, len(dataset)))

    sampled_dataset = Dataset.from_dict({key: [sample[key] for sample in sampled_data] for key in dataset.column_names})

    if remove_columns:
        sampled_dataset = sampled_dataset.remove_columns(remove_columns)

    return sampled_dataset
