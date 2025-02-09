import torch

def metagpt_get_lambdas(tv_list):
    tv_squared_norms = []
    total_squared_norm = 0
    
    for tv_dict in tv_list:
        flattened_tensor = torch.cat([tv_dict[key].view(-1) for key in tv_dict.keys()])
        tv_squared_norm = (torch.norm(flattened_tensor, p=2) ** 2).item()
        tv_squared_norms.append(tv_squared_norm)
        total_squared_norm += tv_squared_norm
    
    lambdas = [tv_squared_norm / total_squared_norm for tv_squared_norm in tv_squared_norms]
    
    return lambdas

def metagpt_get_merged_tv(tv_list):
    merged_tv_dict = {}


    lambdas = metagpt_get_lambdas(tv_list)
    
    for key in tv_list[0].keys():
        merged_tensor = sum(lambdas[i] * tv_list[i][key] for i in range(len(tv_list)))
        merged_tv_dict[key] = merged_tensor
    
    return merged_tv_dict
