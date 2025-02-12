import torch
from tqdm import tqdm
from datasets import load_dataset
from util import random_sample,  clean_model_out


class Mistral_inst_evaluator:
    def __init__(self, dataset_name, cache_dir):
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        self.get_dataset_config()
    
    def get_dataset_config(self):
        print(f'Downloading datasets of {self.dataset_name}')
        if self.dataset_name == 'ethos':
            ds = load_dataset("Lots-of-LoRAs/task1605_ethos_text_classification", cache_dir=self.cache_dir)
            self.num_samples = None
            self.processed_ds = self.process_dataset(ds)
            self.max_new_tokens = 10
            self.text_label_map = {
            '0': ['not violence', 'not_violence', 'non violence', 'non_violence'],
            '1': ['violence'] # the order of 0, 1 here matters
        }
        elif self.dataset_name == 'wino':
            ds = load_dataset("Lots-of-LoRAs/task1391_winogrande_easy_answer_generation", cache_dir=self.cache_dir)
            self.num_samples = 200
            self.processed_ds = self.process_dataset(ds)
            self.max_new_tokens = 10
            self.text_label_map = {
            '1': ['a', 'A'],
            '0': ['b', 'B']
        }
        elif self.dataset_name == 'stereo':
            ds = load_dataset("Lots-of-LoRAs/task280_stereoset_classification_stereotype_type", cache_dir=self.cache_dir)
            self.num_samples = 200
            self.processed_ds = self.process_dataset(ds)
            self.max_new_tokens = 10
            self.text_label_map = {
            '1': ['gender', 'Gender'],
            '2': ['profession', 'Profession'],
            '3': ['race', 'Race'],
            '4': ['religion', 'Religion']
        }   
        elif self.dataset_name == 'causal':
            ds = load_dataset("Lots-of-LoRAs/task391_causal_relationship", cache_dir=self.cache_dir)
            self.num_samples = 200
            self.processed_ds = self.process_dataset(ds)
            self.max_new_tokens = 10
            self.text_label_map = {
            '0': ['not plausible', 'not_plausible', 'non plausible', 'non_plausible'],
            '1': ['plausible'] # the order of 0, 1 here matters
        }
        elif self.dataset_name == 'answerable':
            ds = load_dataset("Lots-of-LoRAs/task290_tellmewhy_question_answerability", cache_dir=self.cache_dir)
            self.num_samples = 200
            self.processed_ds = self.process_dataset(ds)
            self.max_new_tokens = 10
            self.text_label_map = {
            '0': ['not answerable', 'Not Answerable', 'Not_Answerable', 'not_answerable'],
            '1': ['answerable', 'Answerable'] # the order of 0, 1 here matters
        }
        elif self.dataset_name == 'qasc':
            ds = load_dataset("Lots-of-LoRAs/task039_qasc_find_overlapping_words", cache_dir=self.cache_dir)
            self.num_samples = 200
            self.processed_ds = self.process_dataset(ds)
            self.max_new_tokens = 10
        elif self.dataset_name == 'dream':
            ds = load_dataset("Lots-of-LoRAs/task247_dream_answer_generation", cache_dir=self.cache_dir)
            self.num_samples = 200
            self.processed_ds = self.process_dataset(ds)
            self.max_new_tokens = 10
            self.text_label_map = {
            '1': ['(A)', '(a)'],
            '2': ['(B)', '(b)'],
            '3': ['(C)', '(c)']
        }
        elif self.dataset_name == 'ncbi':
            self.num_samples = None
            ds = load_dataset("Lots-of-LoRAs/task1448_disease_entity_extraction_ncbi_dataset", cache_dir=self.cache_dir)
            self.processed_ds = self.process_dataset(ds)
            self.max_new_tokens = 20 
        elif self.dataset_name == 'owant':
            self.num_samples = 200
            ds = load_dataset("Lots-of-LoRAs/task1198_atomic_classification_owant", cache_dir=self.cache_dir)
            self.processed_ds = self.process_dataset(ds)
            self.max_new_tokens = 10 
            self.text_label_map = {
                '1': ['Yes', 'yes'],
                '2': ['No', 'no']
            }
        elif self.dataset_name == 'amazon':
            self.num_samples = 200
            ds = load_dataset("Lots-of-LoRAs/task587_amazonfood_polarity_correction_classification", cache_dir=self.cache_dir)
            self.processed_ds = self.process_dataset(ds)
            self.max_new_tokens = 10 
            self.text_label_map = {
                '1': ['True', 'true'],
                '2': ['False', 'false']
            }
        elif self.dataset_name == 'msr':
            self.num_samples = None
            ds = load_dataset("Lots-of-LoRAs/task1341_msr_text_classification", cache_dir=self.cache_dir)
            self.processed_ds = self.process_dataset(ds)
            self.max_new_tokens = 10 
            self.text_label_map = {
                '1': ['Good', 'good'],
                '2': ['Bad', 'bad']
            }
        elif self.dataset_name == 'gap':
            self.num_samples = 100
            ds = load_dataset("Lots-of-LoRAs/task330_gap_answer_generation", cache_dir=self.cache_dir)
            self.processed_ds = self.process_dataset(ds)
            self.max_new_tokens = 20
        elif self.dataset_name == 'snli':
            self.num_samples = 100
            ds = load_dataset("Lots-of-LoRAs/task190_snli_classification", cache_dir=self.cache_dir)
            self.processed_ds = self.process_dataset(ds)
            self.max_new_tokens = 2
            self.text_label_map = {
                '1': ['C', 'c'],
                '2': ['N', 'n']
            }
        elif self.dataset_name == 'argue':
            self.num_samples = None
            ds = load_dataset("Lots-of-LoRAs/task513_argument_stance_classification", cache_dir=self.cache_dir)
            self.processed_ds = self.process_dataset(ds)
            self.max_new_tokens = 8
            self.text_label_map = {
                '1': ['in favor', 'in_favor', 'In Favor', 'In favor'],
                '2': ['against', 'Against']
            }
        elif self.dataset_name == 'disco':
            self.num_samples = 80
            ds = load_dataset("Lots-of-LoRAs/task564_discofuse_classification", cache_dir=self.cache_dir)
            self.processed_ds = self.process_dataset(ds)
            self.max_new_tokens = 12
            self.text_label_map = {
                '1': ['SINGLE_S_COORD', 'single_s_coord'],
                '2': ['SINGLE_CATAPHORA', 'single_cataphora'],
                '3': ['SINGLE_CONN_INNER', 'single_conn_inner'],
                '4': ['SINGLE_APPOSITION', 'single_apposition'],
                '5': ['SINGLE_VP_COORD', 'single_vp_coord'],
                '6': ['SINGLE_CONN_START', 'single_conn_start'],
                '7': ['PAIR_ANAPHORA', 'pair_anaphora'],
                '8': ['PAIR_CONN', 'pair_conn'],
                '9': ['SINGLE_RELATIVE', 'single_relative'],
                '10': ['SINGLE_CONN_INNER_ANAPHORA', 'single_conn_inner_anaphora'],
                '11': ['SINGLE_S_COORD_ANAPHORA', 'single_s_coord_anaphora'],
                '12': ['PAIR_CONN_ANAPHORA', 'pair_conn_anaphora'],
                '13': ['PAIR_NONE', 'pair_none']
            }
        elif self.dataset_name == 'math':
            self.num_samples = 100
            ds = load_dataset("Lots-of-LoRAs/task834_mathdataset_classification", cache_dir=self.cache_dir)
            self.processed_ds = self.process_dataset(ds)
            self.max_new_tokens = 10
            self.text_label_map = {
                '1': ['Algebra', 'algebra'],
                '2': ['Arithmetic', 'arithmetic'],
                '3': ['Measurement', 'measurement'],
                '4': ['Numbers', 'numbers'],
                '5': ['Probability', 'probability']
            }
        elif self.dataset_name == 'casino':
            self.num_samples = 75
            ds = load_dataset("Lots-of-LoRAs/task357_casino_classification_negotiation_small_talk", cache_dir=self.cache_dir)
            self.processed_ds = self.process_dataset(ds)
            self.max_new_tokens = 10
            self.text_label_map = {
                '1': ['Yes', 'yes'],
                '2': ['No', 'no']
            }
        elif self.dataset_name == 'story':
            self.num_samples = 100
            ds = load_dataset("Lots-of-LoRAs/task298_storycloze_correct_end_classification", cache_dir=self.cache_dir)
            self.processed_ds = self.process_dataset(ds)
            self.max_new_tokens = 3
            self.text_label_map = {
                '1': ['Yes', 'yes'],
                '2': ['No', 'no']
            }
        elif self.dataset_name == 'pubmed':
            self.num_samples = 100
            ds = load_dataset("Lots-of-LoRAs/task846_pubmedqa_classification", cache_dir=self.cache_dir)
            self.processed_ds = self.process_dataset(ds)
            self.max_new_tokens = 3
            self.text_label_map = {
                '1': ['Yes', 'yes'],
                '2': ['No', 'no']
            }
        elif self.dataset_name == 'sst2':
            self.num_samples = 100
            ds = load_dataset("Lots-of-LoRAs/task363_sst2_polarity_classification", cache_dir=self.cache_dir)
            self.processed_ds = self.process_dataset(ds)
            self.max_new_tokens = 3
            self.text_label_map = {
                '1': ['NEG', 'neg', 'negative', 'Negative'],
                '2': ['POS', 'pos', 'positive', 'Positive']
            }
        else:
            raise ValueError('Dataset name not matched')     
            
    
    def process_dataset(self, ds): 
        if self.num_samples is not None: ## ramdom sample
            final_ds = random_sample(ds['test'], num_samples=self.num_samples, remove_columns=['id'])
        else: ## no sample
            final_ds = random_sample(ds['test'], num_samples=len(ds['test']), remove_columns=['id'])       
        return final_ds
    
    def evaluate(self, model, tokenizer, batch_size, device, print_output):
        model.to(device)
        model.eval()
    
        correct_predictions = 0
        total_predictions = 0
        
        questions = [example['input'] for example in self.processed_ds] 
        true_labels = [example['output'][0] for example in self.processed_ds] # the 'output' of Lots of LoRA datasets is formatted in list format

        ## batch inference
        for i in tqdm(range(0, len(questions), batch_size), desc=f"Evaluating given model on {self.dataset_name} dataset"):
            batch_questions = questions[i:i+batch_size]
            batch_true_labels = true_labels[i:i+batch_size]
            batch_messages = [[{"role": "user", "content": q}] for q in batch_questions]
            chats = [tokenizer.apply_chat_template(msg, tokenize=False) for msg in batch_messages]
            encodeds = tokenizer(
                chats,
                padding=True,
                return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                generated_ids = model.generate(
                    **encodeds,
                    pad_token_id=tokenizer.eos_token_id,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False 
                )
            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            for generated_text, true_label in zip(generated_texts, batch_true_labels):
                label_index = generated_text.find("[/INST]")
                if label_index != -1:
                    remaining_text = generated_text[label_index + len("[/INST]"):].strip().lower()
                    if remaining_text:
                        predicted_label = -1
                        hit = False
                        
                        for label_index, labels in self.text_label_map.items():
                            for label in labels:
                                if label in remaining_text:
                                    predicted_label = int(label_index)
                                    hit = True
                                    break
                            if hit:
                                break
                        if hit == False:
                            tqdm.write(f"<Unexpected model outputs: {remaining_text}>") 
                    else:
                        # generate nothing
                        tqdm.write(f"<Nothing is generating>") 
                        predicted_label = -1
                else:
                    raise ValueError("temlate error: '[/INST]' not found in the generated text")
                
                for possible_label, possible_text in self.text_label_map.items():
                    if true_label in possible_text:
                        true_label_id = int(possible_label)
                        break
                if print_output:
                    print('------------------------')
                    print(remaining_text)
                    print(f'True Label is {true_label_id}, Predicted Label is {predicted_label}')
                    print('------------------------')
                
                if predicted_label == true_label_id:
                    correct_predictions += 1
                total_predictions += 1

        accuracy = correct_predictions / total_predictions
        
        # print('cleaning model out from gpu...')
        # clean_model_out(model)

        return accuracy 

    
    def evaluate_F1(self, model, tokenizer, batch_size, device, print_output):       
        model.to(device)
    
        all_precisions = []
        all_recalls = []
        all_f1_scores = []
    
        questions = [example['input'] for example in self.processed_ds]
        true_labels = [example['output'][0] for example in self.processed_ds] # the 'output' of Lots of LoRA datasets is formatted in list format

        ## batch inference
        for i in tqdm(range(0, len(questions), batch_size), desc=f"Evaluating given model on {self.dataset_name} dataset"):
            batch_questions = questions[i:i+batch_size]
            batch_true_labels = true_labels[i:i+batch_size]

            batch_messages = [[{"role": "user", "content": q}] for q in batch_questions]
            chats = [tokenizer.apply_chat_template(msg, tokenize=False) for msg in batch_messages]
            encodeds = tokenizer.batch_encode_plus(
                chats,
                padding=True,
                return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                generated_ids = model.generate(
                    **encodeds,
                    pad_token_id=tokenizer.eos_token_id,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False 
                )
            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            for generated_text, true_label in zip(generated_texts, batch_true_labels):
            
                label_index = generated_text.find("[/INST]")
                if label_index != -1:
                    remaining_text = generated_text[label_index + len("[/INST]"):].strip().lower()
                    predicted_words = set(remaining_text.split())
                else:
                    raise ValueError("Logic error: '[/INST]' not found in the generated text")
        
                true_words = set([word.lower() for word in true_label])
        
                common_words = predicted_words.intersection(true_words)
                precision = len(common_words) / len(predicted_words) if predicted_words else 0
                recall = len(common_words) / len(true_words) if true_words else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
                all_precisions.append(precision)
                all_recalls.append(recall)
                all_f1_scores.append(f1)
        
                if print_output:
                    print('------------------------')
                    print(f'Predicted words: {predicted_words}')
                    print(f'True words: {true_words}')
                    print(f'Precision: {precision}, Recall: {recall}, F1 Score: {f1}')
                    print('------------------------')
    
        avg_precision = sum(all_precisions) / len(all_precisions)
        avg_recall = sum(all_recalls) / len(all_recalls)
        avg_f1_score = sum(all_f1_scores) / len(all_f1_scores)
    
        # print('cleaning model out from gpu...')
        # clean_model_out(model)
    
        # print(f'Average Precision: {avg_precision}')
        # print(f'Average Recall: {avg_recall}')
        # print(f'Average F1 Score: {avg_f1_score}')
    
        return avg_precision, avg_recall, avg_f1_score

    def evaluate_dream(self, model, tokenizer, batch_size, device, print_output):
        model.to(device)
    
        correct_predictions = 0
        total_predictions = 0
        
        questions = [example['input'] for example in self.processed_ds]
        true_labels = [example['output'] for example in self.processed_ds]
        true_labels = [example['output'][0] for example in self.processed_ds] # the 'output' of Lots of LoRA datasets is formatted in list format

        ## batch inference
        for i in tqdm(range(0, len(questions), batch_size), desc=f"Evaluating given model on {self.dataset_name} dataset"):
            batch_questions = questions[i:i+batch_size]
            batch_true_labels = true_labels[i:i+batch_size]
            
            batch_messages = [[{"role": "user", "content": q}] for q in batch_questions]
            chats = [tokenizer.apply_chat_template(msg, tokenize=False) for msg in batch_messages]
            encodeds = tokenizer.batch_encode_plus(
                chats,
                padding=True,
                return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                generated_ids = model.generate(
                    **encodeds,
                    pad_token_id=tokenizer.eos_token_id,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False 
                )
            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            for generated_text, true_label in zip(generated_texts, batch_true_labels):         
            
                label_index = generated_text.find("[/INST]")
                if label_index != -1:
                    remaining_text = generated_text[label_index + len("[/INST]"):].strip().lower()
                    if remaining_text:
                        predicted_label = -1
                        if remaining_text.startswith("(a)") or remaining_text.startswith("a"):
                            predicted_label = 1
                        elif remaining_text.startswith("(b)") or remaining_text.startswith("b"):
                            predicted_label = 2
                        elif remaining_text.startswith("(c)") or remaining_text.startswith("c"):
                            predicted_label = 3
                        else:
                            tqdm.write(f"<Unexpected model outputs: {remaining_text}>") 
                            
                    else:
                        # generate nothing
                        tqdm.write(f"<Nothing is generating>") 
                        predicted_label = -1
                else:
                    raise ValueError("template error: '[/INST]' not found in the generated text")
                
                if true_label.startswith("(A)"):
                    true_label_id = 1
                elif true_label.startswith("(B)"):
                    true_label_id = 2
                elif true_label.startswith("(C)"):
                    true_label_id = 3
                
                if print_output:
                    print('------------------------')
                    print(remaining_text)
                    print(f'True Label is {true_label_id}, Predicted Label is {predicted_label}')
                    print('------------------------')
                
                if predicted_label == true_label_id:
                    correct_predictions += 1
                total_predictions += 1

        accuracy = correct_predictions / total_predictions
        
        # print('cleaning model out from gpu...')
        # clean_model_out(model)

        return accuracy 

    def evaluate_ncbi_gap(self, model, tokenizer, batch_size, device, print_output):
        model.to(device)
    
        correct_predictions = 0
        total_predictions = 0
        
        questions = [example['input'] for example in self.processed_ds]
        true_labels = [example['output'][0] for example in self.processed_ds] # the 'output' of Lots of LoRA datasets is formatted in list format

        
        ## batch inference
        for i in tqdm(range(0, len(questions), batch_size), desc=f"Evaluating given model on {self.dataset_name} dataset"):
            batch_questions = questions[i:i+batch_size]
            batch_true_labels = true_labels[i:i+batch_size]
            
            batch_messages = [[{"role": "user", "content": q}] for q in batch_questions]
            chats = [tokenizer.apply_chat_template(msg, tokenize=False) for msg in batch_messages]
            encodeds = tokenizer.batch_encode_plus(
                chats,
                padding=True,
                return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                generated_ids = model.generate(
                    **encodeds,
                    pad_token_id=tokenizer.eos_token_id,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False 
                )
            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            for generated_text, true_label in zip(generated_texts, batch_true_labels):
                label_index = generated_text.find("[/INST]")
                if label_index != -1:
                    remaining_text = generated_text[label_index + len("[/INST]"):].strip().lower()
                    if remaining_text:
                        predicted_label = -1
                        if true_label.lower() in remaining_text:
                            predicted_label = 1
                        else:
                            tqdm.write(f"<Unexpected model outputs: {remaining_text}>")
                    else:
                        # generate nothing
                        tqdm.write(f"<Nothing is generating>") 
                        predicted_label = -1
                else:
                    raise ValueError("template error: '[/INST]' not found in the generated text")
                            
                if print_output:
                    print('------------------------')
                    print(remaining_text)
                    print(f'True Label is {true_label}, Predicted Label is {predicted_label}')
                    print('------------------------')
                
                if predicted_label == 1:
                    correct_predictions += 1
                total_predictions += 1

        accuracy = correct_predictions / total_predictions
        
        # print('cleaning model out from gpu...')
        # # clean_model_out(model)

        return accuracy 