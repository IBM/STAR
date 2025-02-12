import torch
from tqdm import tqdm

def global_trim_weights(tv_dicts, k, device):
    trimmed_tv_dicts = {}

    for task, tv_dict in tv_dicts.items():
        print(f'========= Trim {task} =========')
        
        tv_dict_gpu = {key: tensor.to(device) for key, tensor in tv_dict.items()} # move to gpu for later fast 'torch.topk'
        all_values = torch.cat([tv_dict_gpu[key].view(-1) for key in tqdm(tv_dict_gpu.keys(), desc='Flatten')])
        
        keep_num = int(all_values.size(0) * k / 100)
        threshold_value = torch.topk(all_values.abs(), keep_num, largest=True)[0][-1]

        trimmed_tv_dict = {}

        for key in tqdm(tv_dict_gpu.keys(), desc="Trim"):
            tensor = tv_dict_gpu[key]
            trimmed_tensor = torch.where(tensor.abs() >= threshold_value, tensor, torch.tensor(0.0, device=device))
            trimmed_tv_dict[key] = trimmed_tensor.to("cpu")
        trimmed_tv_dicts[task] = trimmed_tv_dict
        del tv_dict_gpu, all_values, threshold_value
        torch.cuda.empty_cache()
    return trimmed_tv_dicts

def local_trim_weights(tv_dicts, k, device):
    trimmed_tv_dicts = {}

    for task, tv_dict in tv_dicts.items():
        print(f'========= Trim {task} =========')
        trimmed_tv_dict = {}

        tv_dict_gpu = {key: tensor.to(device) for key, tensor in tv_dict.items()} # move to gpu for later fast 'torch.topk'

        for key in tqdm(tv_dict_gpu.keys(), desc="Trim"):
            tensor = tv_dict_gpu[key]
            num_elements = tensor.numel()
            keep_num = int(num_elements * k / 100)
            threshold_value = torch.topk(tensor.abs().view(-1), keep_num, largest=True)[0][-1]
            trimmed_tensor = torch.where(tensor.abs() >= threshold_value, tensor, torch.tensor(0.0, device=device))
            trimmed_tv_dict[key] = trimmed_tensor.to("cpu")

        trimmed_tv_dicts[task] = trimmed_tv_dict

        del tv_dict_gpu
        torch.cuda.empty_cache()

    return trimmed_tv_dicts


def elect_sign(trimmed_tv_dicts):
    elect_sign_dict = {}

    layer_keys = next(iter(trimmed_tv_dicts.values())).keys()

    for key in layer_keys:
        summed_matrix = sum(trimmed_tv_dicts[task][key] for task in trimmed_tv_dicts)
        elect_sign_dict[key] = torch.sign(summed_matrix)

    return elect_sign_dict

def disjoint_merge(trimmed_tv_dicts, sign_reference_dict):
    merged_tv_dict = {}

    layer_keys = next(iter(trimmed_tv_dicts.values())).keys()

    for key in layer_keys:
        # print(f'Merging {key}')

        sign_reference = sign_reference_dict[key]
        
        accumulated_matrix = torch.zeros_like(sign_reference, dtype=torch.float)
        match_count = torch.zeros_like(sign_reference, dtype=torch.float)
        
        # Iterate over each task vector
        for trimmed_tv_dict in trimmed_tv_dicts.values():
            # Create a matching mask that represents sign agreement beween trimmed task vector and elect sign result
            matching_mask = (torch.sign(trimmed_tv_dict[key]) == sign_reference) & (torch.sign(trimmed_tv_dict[key]) != 0)

            # Accumulate matching parameters
            accumulated_matrix += torch.where(matching_mask, trimmed_tv_dict[key], torch.tensor(0.0))

            # Update the match count
            match_count += matching_mask.int()
                   
        # Calculate the average of matched parameters
        averaged_matrix = torch.zeros_like(accumulated_matrix)
        non_zero_mask = match_count > 0 # To avoid division by zero,  explicitly handle match_count that is not zero only
        averaged_matrix[non_zero_mask] = accumulated_matrix[non_zero_mask] / match_count[non_zero_mask].float()
        
        # Store the merged result in the dictionary
        merged_tv_dict[key] = averaged_matrix

        # merged_tv_dict[key] = averaged_matrix * scaling_factor ## do not use scaling factor, since it would require data tuning on validation set 
    return merged_tv_dict
