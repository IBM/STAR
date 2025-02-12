import torch
from tqdm import tqdm

def star_task_vectors(tv_dicts, eta, known_rank=None):
    star_tv_dicts = {}
    for task, tv_dict in tv_dicts.items():
        print(f'========= STAR on {task} =========')
        star_tv_dict = {}
        
        for key, weight_matrix in tqdm(tv_dict.items(), desc="SVD"):
            star_tv_dict[key] =  star(weight_matrix, eta, known_rank)        
        star_tv_dicts[task] = star_tv_dict
    return star_tv_dicts


def star(matrix, eta, known_rank=None):
    if known_rank is not None:
        '''
         If matrix rank is known in advanced, we can do lower rank SVD for faster computation 
         https://pytorch.org/docs/stable/generated/torch.svd_lowrank.html
        '''
        u, s, v = torch.svd_lowrank(matrix, q=known_rank+5)
        s = s[:known_rank]
    else:
        u, s, vt = torch.linalg.svd(matrix, full_matrices=False)
        
    # Compute cumulative sum to determine rank_remain
    sum_tot_s = torch.sum(s)
    cumulative_tot_s = torch.cumsum(s, dim=0)
    rank_remain = torch.searchsorted(cumulative_tot_s, sum_tot_s * eta / 100).item() + 1

    
    # print(f'{rank_remain} is enough for keeping {eta}% nuclear norm')

    # Truncate U, V, and S
    u_truncated = u[:, :rank_remain]
    if known_rank is not None:
        v_truncated = v[:, :rank_remain]
    else:
        vt_truncated = vt[:rank_remain, :]
    remain_s = s[:rank_remain]

    # Scale the retained singular values
    sum_remain_s = torch.sum(remain_s)
    scaled_remain_s = (sum_tot_s / sum_remain_s) * remain_s

    # Reconstruct the truncated matrix
    if known_rank is not None:
        truncated_matrix = u_truncated @ torch.diag_embed(scaled_remain_s) @ v_truncated.T
    else:
        truncated_matrix = u_truncated @ torch.diag_embed(scaled_remain_s) @ vt_truncated
    
    return truncated_matrix

