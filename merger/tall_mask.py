import torch

def get_merged_tv_tall(tv_list, lambda_, scaling_factor, k=2):
    merged_tv_dict = {}
    
    mtl_dict = get_mtl(tv_list, scaling_factor) # i.e., Task Arithmetic
    
    tall_mask_dicts = get_tall_masks(tv_list, mtl_dict, lambda_)
    
    cs_mask_dict = consensus_mask(tall_mask_dicts, k)

    merged_tv_dict = get_merged_tv(mtl_dict, cs_mask_dict)

    return merged_tv_dict


def get_mtl(tv_list, scaling_factor):
    mtl_dict = {}

    ### Apply Task Arithmeitc Here, for "Consensus Task Arithmetic"
    for layer in tv_list[0].keys():
        # print(layer)
        mtl = torch.zeros(tv_list[0][layer].shape)
        for tv_dict in tv_list:
            mtl += tv_dict[layer]
        
        mtl_dict[layer] = mtl * scaling_factor
    return mtl_dict

def get_tall_masks(tv_list, mtl_dict, lambda_):
    tall_mask_dicts = []
    for tv_dict in tv_list:
        tall_mask_dict = {}
        for layer, tensor in tv_dict.items():
            tall_mask_dict[layer] = tensor.abs() > (mtl_dict[layer] - tensor).abs() * lambda_
        tall_mask_dicts.append(tall_mask_dict)
    return tall_mask_dicts

def consensus_mask(tall_mask_dicts, k):
    cs_mask_dict = {}    
    for layer, tensor in tall_mask_dicts[0].items():
        agree = torch.zeros(tensor.shape)
        for tall_mask_dict in tall_mask_dicts:
            agree += tall_mask_dict[layer].int()
        cs_mask_dict[layer] = (agree >= k).int()
    return cs_mask_dict

def get_merged_tv(mtl_dict, cs_mask_dict):
    merged_tv_dict = {}
    for layer in mtl_dict.keys():
        merged_tv_dict[layer] = cs_mask_dict[layer] * mtl_dict[layer]
    return merged_tv_dict