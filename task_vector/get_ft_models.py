from peft import PeftConfig, load_peft_weights
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


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