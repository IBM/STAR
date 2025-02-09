import torch

def weighted_merge(lora_base_form_task_vectors_list, alphas):
       
    merged_base_form_task_vector = {}

    # Merge the weight matrices
    for base_name in lora_base_form_task_vectors_list[0].keys():
        merged_parameter = None
        for lora_idx, lora_base_form_task_vector in enumerate(lora_base_form_task_vectors_list):
            parameter = lora_base_form_task_vector[base_name]
            
            if merged_parameter is None:
                merged_parameter = torch.clone(parameter) * alphas[lora_idx]
            else:
                merged_parameter += parameter * alphas[lora_idx]
    
        merged_base_form_task_vector[base_name] = merged_parameter

    return merged_base_form_task_vector