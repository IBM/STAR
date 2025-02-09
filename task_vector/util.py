import gc
import torch
import copy
from tqdm import tqdm

def load_lora_into_base(base_model, lora_parameter_base_form_dict):
    base_model_for_lora = copy.deepcopy(base_model)
    base_state_dict = base_model_for_lora.cpu().state_dict()
    
    for base_name, lora_parameter in tqdm(lora_parameter_base_form_dict.items(), desc="Loading"):
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

