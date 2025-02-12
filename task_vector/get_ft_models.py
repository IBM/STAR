from peft import PeftConfig, load_peft_weights
from transformers import AutoModelForSeq2SeqLM


def get_ft_model_tv(base_model, model_name, target_keys, cache_dir):
    '''
    Calculate task vectors for LoRAs that were in base model form
    '''
    ft_model = AutoModelForSeq2SeqLM.from_pretrained(model_name,  cache_dir = cache_dir)
    tv_dict = {}

    for key, ft_weight in ft_model.state_dict().items():
        if key in target_keys:
            tv_dict[key] = ft_weight - base_model.state_dict()[key]
    return tv_dict


def load_lora(model_name, cache_dir):
    lora_model_parameters_dict = load_peft_weights(model_name, cache_dir=cache_dir)
    for key, value in lora_model_parameters_dict.items():
        lora_model_parameters_dict[key] = value.to('cpu')

    lora_rank = PeftConfig.from_pretrained(model_name, cache_dir=cache_dir).r
    lora_alpha = PeftConfig.from_pretrained(model_name, cache_dir=cache_dir).lora_alpha
    return lora_rank, lora_alpha, lora_model_parameters_dict

def transform_lora_dict_to_base_form(lora_model_parameters_dict, lora_rank, lora_alpha):
    lora_parameter_base_form_dict = {}
    for parameter_name in lora_model_parameters_dict.keys():
        if "lora_A" in parameter_name:
            # print(f'handling {parameter_name}')
            lora_A = lora_model_parameters_dict[parameter_name]
            lora_B = lora_model_parameters_dict[parameter_name.replace("lora_A", "lora_B")]
    
            base_name = parameter_name.replace("lora_A.weight", "weight")
            base_name = base_name.replace("base_model.model.", "")
    
            product = lora_B @ lora_A  # Merge the B, A
    
            lora_parameter_base_form_dict[base_name] = product * (lora_alpha/lora_rank) # important to scale LoRA's parameters this way
    return lora_parameter_base_form_dict


def load_flan_t5_base_ft_tvs(tasks, cache_dir):
    tv_dicts = {}
    print(f'Downloading LoRAs of Flan-T5-base, to {cache_dir}')
    for task in tasks:
        print(task)
        model_name = f"tanganke/flan-t5-base_glue-{task}_lora-16"
        lora_alpha, lora_parameters_dict = load_lora(model_name, cache_dir)

        ## Multiply LoRA's B*A, and transform key name to fit the form of base model
        lora_parameters_base_form_dict  = transform_lora_dict_to_base_form(lora_parameters_dict, lora_alpha)
        
        tv_dicts[task] = lora_parameters_base_form_dict
    return tv_dicts

def load_flan_t5_large_ft_tvs(base_model, tasks, cache_dir):
    tv_dicts = {}
    print(f'Downloading LoRAs of Flan-T5-large, to {cache_dir}')
    for task in tasks:
        print(task)
        if task in ['mnli', 'mrpc', 'qnli', 'qnli', 'qqp', 'rte', 'sst2', 'stsb']:
            model_name = f"tanganke/flan-t5-large_glue-{task}_lora-16"
            lora_rank, lora_alpha, lora_parameters_dict = load_lora(model_name, cache_dir)
            
            ## Multiply LoRA's B*A, and transform key name to fit the form of base model
            tv_dict  = transform_lora_dict_to_base_form(lora_parameters_dict, lora_rank, lora_alpha)
    
        elif task in ['finance', 'imdb', 'agnews', 'hella', 'boolq', 'piqa']: ## These LoRAs are already merged into base model
            target_keys = ['decoder.block.{}.layer.0.SelfAttention.q.weight'.format(i) for i in range(24)] + \
                  ['decoder.block.{}.layer.0.SelfAttention.v.weight'.format(i) for i in range(24)] + \
                  ['decoder.block.{}.layer.1.EncDecAttention.q.weight'.format(i) for i in range(24)] + \
                  ['decoder.block.{}.layer.1.EncDecAttention.v.weight'.format(i) for i in range(24)] + \
                  ['encoder.block.{}.layer.0.SelfAttention.q.weight'.format(i) for i in range(24)] + \
                  ['encoder.block.{}.layer.0.SelfAttention.v.weight'.format(i) for i in range(24)]
            
            model_name = f"Speeeed/flan-t5-large-{task}_lora-16"
            tv_dict = get_ft_model_tv(base_model, model_name, target_keys, cache_dir)
        else:
            raise ValueError(f'{task} fine-tuned model currently not support')
        
        tv_dicts[task] = tv_dict
    return tv_dicts

def load_mistral_inst_ft_tvs(base_model, tasks, cache_dir):
    tv_dicts = {}
    print(f'Downloading LoRAs of Mistral_instruct, to {cache_dir}')
    ## link the task name we set to actual model id in Lots of LoRAs
    task_id_map = {
    'ethos': 1605,
    'wino': 1391,
    'stereo': 280,
    'causal': 391,
    'answerable': 290,
    'qasc': 39,
    'dream': 247,
    'ncbi': 1448,
    'owant': 1198,
    'amazon': 587,
    'msr': 1341,
    'gap': 330,
    'snli': 190,
    'argue': 513,
    'disco': 564,
    'math': 834,
    'casino': 357,
    'story': 298,
    'pubmed': 846,
    'sst2': 363
    } 
    for task in tasks:
        print(task)
        if task in ['ethos', 'wino', 'stereo', 'causal', 'answerable', 'qasc', 'dream', 'ncbi', 'owant', 'amazon', 'msr', 'gap', 'snli', 'argue', 'disco', 'math', 'casino', 'story', 'pubmed', 'sst2']:
            model_name = f"Lots-of-LoRAs/Mistral-7B-Instruct-v0.2-4b-r16-task{task_id_map[task]}"
            lora_rank, lora_alpha, lora_parameters_dict = load_lora(model_name, cache_dir)
            ## Multiply LoRA's B*A, and transform key name to fit the form of base model
            tv_dict  = transform_lora_dict_to_base_form(lora_parameters_dict, lora_rank, lora_alpha)
        else:
            raise ValueError(f'{task} fine-tuned model currently not support')
        
        tv_dicts[task] = tv_dict
    return tv_dicts


def load_llama_3b_ft_tvs(base_model, tasks, cache_dir):
    tv_dicts = {}
    print(f'Downloading LoRAs of Llama_3b, to {cache_dir}')
    for task in tasks:
        print(task)
        if task in ['sst2', 'mrpc', 'wic', 'cola', 'mnli', 'stsb', 'cb', 'multirc', 'boolq', 'rte', 'copa', 'wsc', 'qnli', 'qqp']:
            model_name = f"Speeeed/Llama-3.2-3B-instruct-{task}-lora"
            lora_rank, lora_alpha, lora_parameters_dict = load_lora(model_name, cache_dir)
            ## Multiply LoRA's B*A, and transform key name to fit the form of base model
            tv_dict  = transform_lora_dict_to_base_form(lora_parameters_dict, lora_rank, lora_alpha)
        else:
            raise ValueError(f'{task} fine-tuned model currently not support')
        
        tv_dicts[task] = tv_dict
    return tv_dicts