import torch
from tqdm import tqdm

def star_task_vectors(tv_dicts, eta):
    star_tv_dicts = {}
    for task, tv_dict in tv_dicts.items():
        print(f'========= STAR on {task} =========')
        star_tv_dict = {}
        
        for key, weight_matrix in tqdm(tv_dict.items(), desc="SVD"):
            star_tv_dict[key] =  star(weight_matrix, eta)        
        star_tv_dicts[task] = star_tv_dict
    return star_tv_dicts


def star(matrix, eta):
    u, s, v = torch.linalg.svd(matrix)
    s = s[:16]

    # Compute cumulative sum to determine rank_remain
    sum_tot_s = torch.sum(s)
    cumulative_tot_s = torch.cumsum(s, dim=0)
    rank_remain = torch.searchsorted(cumulative_tot_s, sum_tot_s * eta / 100).item() + 1

    
    # print(f'{rank_remain} is enough for keeping {eta}% Lp norm')

    # Truncate U, V, and S
    u_truncated = u[:, :rank_remain]
    v_truncated = v[:rank_remain, :]  
    
    remain_s = s[:rank_remain]

    # Scale the retained singular values
    sum_remain_s = torch.sum(remain_s)
    scaled_remain_s = (sum_tot_s / sum_remain_s) * remain_s

    # Reconstruct the truncated matrix
    truncated_matrix = u_truncated @ torch.diag_embed(scaled_remain_s) @ v_truncated

    return truncated_matrix

