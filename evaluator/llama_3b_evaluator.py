from tqdm import tqdm
from datasets import load_dataset, Dataset
from collections import Counter
from util import random_balanced_sample,  clean_model_out
import torch


class Llama_3_2_instruct_evaluator:
    def __init__(self, dataset_name, cache_dir):
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        self.get_dataset_config()

    def get_dataset_config(self):
        print(f'Downloading datasets of {self.dataset_name}')
        if self.dataset_name == 'multirc':
            ds = load_dataset("aps/super_glue", "multirc", cache_dir=self.cache_dir)  
            self.num_samples_per_label = 150          
            self.processed_ds = self.process_dataset(ds)
            self.max_new_tokens = 10
            self.text_label_map = {
                '0': ['no', 'No'],
                '1': ['yes', 'Yes'],
            }
        elif self.dataset_name == 'wic':
            ds = load_dataset("aps/super_glue", "wic", cache_dir=self.cache_dir)            
            self.num_samples_per_label = 150
            self.processed_ds = self.process_dataset(ds)
            self.max_new_tokens = 10
            self.text_label_map = {
                '0': ['no', 'No'],
                '1': ['yes', 'Yes'],
            }
        elif self.dataset_name == 'wsc':
            ds = load_dataset("aps/super_glue", "wsc", cache_dir=self.cache_dir)            
            self.num_samples_per_label = 150
            self.processed_ds = self.process_dataset(ds)
            self.max_new_tokens = 10
            self.text_label_map = {
                            '1': ['yes', 'Yes'],
                            '0': ['no', 'No'],
                        }
        elif self.dataset_name == 'cb':
            ds = load_dataset("aps/super_glue", "cb", cache_dir=self.cache_dir)            
            self.processed_ds = self.process_dataset(ds)
            self.max_new_tokens = 10
            self.text_label_map = {
                            '0': ['yes', 'Yes'],
                            '1': ['no', 'No'],
                            '2': ['maybe', 'Maybe']
                        }
        elif self.dataset_name == 'cola':
            ds = load_dataset("nyu-mll/glue", "cola", cache_dir=self.cache_dir)
            self.num_samples_per_label = 150
            self.processed_ds = self.process_dataset(ds)
            self.max_new_tokens = 10
            self.text_label_map = {
                '0': ['no', 'No'],
                '1': ['yes', 'Yes'],                
            }
        elif self.dataset_name == 'mrpc':
            ds = load_dataset("nyu-mll/glue", "mrpc", cache_dir=self.cache_dir)
            self.num_samples_per_label = 150
            self.processed_ds = self.process_dataset(ds)
            self.max_new_tokens = 10
            self.text_label_map = {
                '0': ['no', 'No'],
                '1': ['yes', 'Yes'],

            }
        elif self.dataset_name == 'sst2':
            ds = load_dataset("nyu-mll/glue", "sst2", cache_dir=self.cache_dir)
            self.num_samples_per_label = 150
            self.processed_ds = self.process_dataset(ds)
            self.max_new_tokens = 10
            self.text_label_map = {
                '1': ['positive', 'Positive'],
                '0': ['negative', 'Negative'],
            }
        elif self.dataset_name == 'qqp':
            ds = load_dataset("nyu-mll/glue", "qqp", cache_dir=self.cache_dir)
            self.num_samples_per_label = 150
            self.processed_ds = self.process_dataset(ds)
            self.max_new_tokens = 10
            self.text_label_map = {
                '1': ['yes', 'Yes'],
                '0': ['no', 'No'],
            }
        elif self.dataset_name == 'qnli':
            ds = load_dataset("nyu-mll/glue", "qnli", cache_dir=self.cache_dir)
            self.num_samples_per_label = 150
            self.processed_ds = self.process_dataset(ds)
            self.max_new_tokens = 10
            self.text_label_map = {
                '1': ['false', 'False'],
                '0': ['true', 'True'],
            }
        elif self.dataset_name == 'mnli':
            ds = load_dataset("nyu-mll/glue", "mnli_matched", cache_dir=self.cache_dir)
            self.num_samples_per_label = 150
            self.processed_ds = self.process_dataset(ds)
            self.max_new_tokens = 10
            self.text_label_map = {
                '0': ['entail', 'Entail', 'entailment', 'Entailment'],
                '1': ['neither', 'Neither'],
                '2': ['contradiction', 'Contradiction', 'contradict', 'Contradict'] ## 偷改過(for PEFT有較好結果)
            }
        elif self.dataset_name == 'copa':
            ds = load_dataset("aps/super_glue", "copa", cache_dir=self.cache_dir)            
            self.num_samples_per_label = 150
            self.processed_ds = self.process_dataset(ds)
            self.max_new_tokens = 10
            self.text_label_map = {
                            '1': ['choice2', 'Choice2'],
                            '0': ['choice1', 'Choice1'],
                        }
        elif self.dataset_name == 'rte':
            ds = load_dataset("nyu-mll/glue", "rte", cache_dir=self.cache_dir)
            self.num_samples_per_label = 150
            self.processed_ds = self.process_dataset(ds)
            self.max_new_tokens = 10
            self.text_label_map = {
                '1': ['no', 'No'],
                '0': ['yes', 'Yes'],
            }
        elif self.dataset_name == 'stsb':
            ds = load_dataset("nyu-mll/glue", "stsb", cache_dir=self.cache_dir)
            self.processed_ds = self.process_dataset(ds)
            self.max_new_tokens = 10
        else:
            raise ValueError(f"Dataset {self.dataset_name} is not supported.")

    
    def process_dataset(self, ds): 
        if self.dataset_name in ['sst2', 'wic', 'cola', 'mnli', 'multirc', 'boolq', 'rte', 'copa', 'wsc', 'qnli', 'qqp']:
            final_ds = random_balanced_sample(ds['validation'], label_column='label', num_samples_per_label=self.num_samples_per_label)
        elif self.dataset_name in ['mrpc']:
            final_ds = random_balanced_sample(ds['test'], label_column='label', num_samples_per_label=self.num_samples_per_label)
        elif self.dataset_name in ['cb']:
            final_ds = ds['validation']
        elif self.dataset_name in ['stsb']:
            import random
            from torch.utils.data import Subset
            random.seed(42)
            validation_ds = ds['validation']
            num_samples = 200
            indices = list(range(len(validation_ds)))
            random_indices = random.sample(indices, num_samples)
            final_ds = Subset(validation_ds, random_indices)  
        else:
            raise ValueError('Dataset name not matched')
        return final_ds

    def evaluate(self, model, tokenizer, device, batch_size, print_output=False):
        if batch_size is not None:
            print('batch infernce is not implemented for Llama, return to single inference')
        model.to(device)

        # print(f"Padding side: {tokenizer.padding_side}")
        
        correct_predictions = 0
        total_predictions = 0
        
        if self.dataset_name == 'cola':
            sentences = [example['sentence'] for example in self.processed_ds]
            true_labels = [example['label'] for example in self.processed_ds]
            messages = [
                [{"role": "user", "content": f"Indicate if the following sentence is grammatically correct or not: \"{sentence}\". Answer ‘Yes’ or ‘No’."}]
                for sentence in sentences]
        elif self.dataset_name == 'mrpc':
            sentence1s = [example['sentence1'] for example in self.processed_ds]
            sentence2s = [example['sentence2'] for example in self.processed_ds]
            true_labels = [example['label'] for example in self.processed_ds]
            messages = [
                [{"role": "user", "content": f"Do these two sentences convey the same meaning? Answer only 'Yes' or 'No'. sentence1: {sentence1}, sentence2: {sentence2}"}]
                for sentence1, sentence2 in zip(sentence1s, sentence2s)]    
        elif self.dataset_name == 'wic':
            words = [example['word'] for example in self.processed_ds]
            sentence1s = [example['sentence1'] for example in self.processed_ds]
            sentence2s = [example['sentence2'] for example in self.processed_ds]
            true_labels = [example['label'] for example in self.processed_ds]
            messages = [
                [{"role": "user", "content": f"For the word '{word}' in these two sentences: '{sentence1}' and '{sentence2}', does it have the same meaning? Answer only 'Yes' or 'No'."}]
                for  word, sentence1, sentence2 in zip(words, sentence1s, sentence2s) 
            ]
        elif self.dataset_name == 'sst2':
            sentences = [example['sentence'] for example in self.processed_ds]
            true_labels = [example['label'] for example in self.processed_ds]
            messages = [
                [{"role": "user", "content": f"Is the sentiment expressed in this text positive or negative? text: '{sentence}' Answer with 'Positive' or 'Negative'"}]
                for  sentence in sentences 
            ]
        elif self.dataset_name == 'multirc':
            paragraphs = [example['paragraph'] for example in self.processed_ds]
            answers = [example['answer'] for example in self.processed_ds]
            questions = [example['question'] for example in self.processed_ds]
            true_labels = [example['label'] for example in self.processed_ds]
            messages = [
                [{"role": "user", "content": f"Based on this paragraph: '{paragraph}'\n\nIs the answer: '{answer}' correct for the question: '{question}'?\n\nAnswer only 'Yes' or 'No'."}]
                for  paragraph, answer, question in zip(paragraphs, answers, questions) 
            ]
        elif self.dataset_name == 'qnli':
            sentences = [example['sentence'] for example in self.processed_ds]
            questions = [example['question'] for example in self.processed_ds]
            true_labels = [example['label'] for example in self.processed_ds]
            messages = [
                [{"role": "user", "content": f"Based on this sentence: '{sentence}', does it answer the question: '{question}'? Answer with 'True' or 'False'."}]
                for  sentence, question in zip(sentences, questions) 
            ]
        elif self.dataset_name == 'mnli':
            premises = [example['premise'] for example in self.processed_ds]
            hypothesiss = [example['hypothesis'] for example in self.processed_ds]
            true_labels = [example['label'] for example in self.processed_ds]
            messages = [
                [{"role": "user", "content": f"Read these two statements:\n\nStatement 1: '{premise}'\nStatement 2: '{hypothesis}'\n\nDoes Statement 1 entail, contradict, or neither for Statement 2? Answer only with 'Entail', 'Contradict', or 'Neither', do not explain."}]
                for  premise, hypothesis in zip(premises, hypothesiss) 
            ]
        elif self.dataset_name == 'copa':
            premises = [example['premise'] for example in self.processed_ds]
            questions = [example['question'] for example in self.processed_ds]
            choice1s = [example['choice1'] for example in self.processed_ds]
            choice2s = [example['choice2'] for example in self.processed_ds]
            true_labels = [example['label'] for example in self.processed_ds]
            messages = [
                [{"role": "user", "content": f"Given the premise: '{premise}', what's the {question} for this? choice1: '{choice1}' or choice2: '{choice2}'. Answer with 'choice1' or 'choice2' without explanation."}]
                for  premise, question, choice1, choice2 in zip(premises, questions, choice1s, choice2s) 
            ]
        elif self.dataset_name == 'rte':
            sentence1s = [example['sentence1'] for example in self.processed_ds]
            sentence2s = [example['sentence2'] for example in self.processed_ds]
            true_labels = [example['label'] for example in self.processed_ds]
            messages = [
                [ {"role": "user", "content": f"Does the text: ‘{sentence1}’ entail that ‘{sentence2}’ is true? Provide ‘yes’ or ‘no’."}]
                for  sentence1, sentence2 in zip(sentence1s, sentence2s) 
            ]
        elif self.dataset_name == 'wsc':
            texts = [example['text'] for example in self.processed_ds]
            span2_texts = [example['span2_text'] for example in self.processed_ds]
            span1_texts = [example['span1_text'] for example in self.processed_ds]
            true_labels = [example['label'] for example in self.processed_ds]
            messages = [
                [{"role": "user", "content": f"In the sentence: '{text}'\nDoes the word '{span2_text}' refer to '{span1_text}'? Answer only 'Yes' or 'No'."}]
                for  text, span2_text, span1_text in zip(texts, span2_texts, span1_texts) 
            ]
        elif self.dataset_name == 'cb':
            premises = [example['premise'] for example in self.processed_ds]
            hypothesiss = [example['hypothesis'] for example in self.processed_ds]
            true_labels = [example['label'] for example in self.processed_ds]
            messages = [
                [{"role": "user", "content": f"Given the following premise:\n'{premise}'\n\nDoes the hypothesis:\n'{hypothesis}'\n\nappear to be true based on the premise?\nAnswer with 'yes', 'no', or 'maybe' without explanation."}]
                for  premise, hypothesis in zip(premises, hypothesiss) 
            ]
        elif self.dataset_name == 'qqp':
            question1s = [example['question1'] for example in self.processed_ds]
            question2s = [example['question2'] for example in self.processed_ds]
            true_labels = [example['label'] for example in self.processed_ds]
            messages = [
                [{"role": "user", "content": f"Do the questions ‘{question1}’ and ‘{question2}’ have the same intent? Answer with ‘yes’ or ‘no’."}]
                for  question1, question2 in zip(question1s, question2s) 
            ]
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        for message, true_label in tqdm(zip(messages, true_labels), total=len(self.processed_ds), desc=f"Evaluating {self.dataset_name} dataset"):
            formatted_prompt = tokenizer.apply_chat_template(
                message,
                add_generation_prompt=True,
                tokenize=False
            )

            inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True).to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id, # for supress warning
                    eos_token_id=terminators, 
                    do_sample=False, 
                    top_p=1.0, # for supress warning
                    temperature=1.0 # for supress warning
                )

            response = outputs[0][inputs['input_ids'].shape[-1]:]
            generated_text = tokenizer.decode(response, skip_special_tokens=True)

            predicted_label = -1
            hit = False
            
            if generated_text:
                for label_index, labels in self.text_label_map.items():
                    for label in labels:
                        if label.lower() in generated_text.lower():
                            predicted_label = int(label_index)
                            hit = True
                            break
                    if hit:
                        break
                if not hit:
                    tqdm.write(f"<Unexpected model outputs: {generated_text}>")                 
            else:
                tqdm.write(f"<Nothing is generating>") 
            
            if print_output:
                print('------------------------')
                print(f"Generated Text: {generated_text}")
                print(f"True Label: {true_label}, Predicted Label: {predicted_label}")
                print('------------------------')
            
            if predicted_label == int(true_label):
                correct_predictions += 1
            total_predictions += 1
        
        accuracy = correct_predictions / total_predictions
        return accuracy



    def evaluate_stsb(self, model, tokenizer, device, batch_size, print_output=False):      
        import re
        from scipy.stats import spearmanr
        model.to(device)

        predicted_scores = []
        ground_truth_scores = []

        if self.dataset_name == 'stsb':
            sentence1s = [example['sentence1'] for example in self.processed_ds]
            sentence2s = [example['sentence2'] for example in self.processed_ds]
            true_labels = [example['label'] for example in self.processed_ds]
            messages = [
                [ {"role": "user", "content": f"Consider the sentences '{sentence1}' and '{sentence2}'. On a scale from 1 (completely different) to 5 (completely similar), rate the similarity. Return only a number with one decimal place, no explanation."}]
                for  sentence1, sentence2 in zip(sentence1s, sentence2s) 
            ]
        else:
            raise ValueError(f'do not used evaluate_stsb for {self.dataset_name}')
        

        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        for message, true_label in tqdm(zip(messages, true_labels), total=len(self.processed_ds), desc=f"Evaluating {self.dataset_name} dataset"):
            formatted_prompt = tokenizer.apply_chat_template(
                message,
                add_generation_prompt=True,
                tokenize=False
            )

            inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True).to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id, # for supress warning
                    eos_token_id=terminators, 
                    do_sample=False, 
                    top_p=1.0, # for supress warning
                    temperature=1.0 # for supress warning
                )

            response = outputs[0][inputs['input_ids'].shape[-1]:]
            generated_text = tokenizer.decode(response, skip_special_tokens=True)

        
            match = re.search(r"[-+]?\d*\.\d+|\d+", generated_text)
            if match:
                try:
                    predicted_score = float(match.group())
                    predicted_score = max(0, min(5, predicted_score))
                except ValueError:
                    tqdm.write(f"<Unable to convert generated text to float: {generated_text}>")
                    predicted_score = 0.0
            else:
                tqdm.write(f"<No valid number found in generated text: {generated_text}>")
                predicted_score = 0.0
                
            predicted_scores.append(predicted_score)
            rounded_true_label = round(true_label, 1)
            ground_truth_scores.append(rounded_true_label)
                
            if print_output:
                print('------------------------')
                print(f"Generated Text: {generated_text}")
                print(f"True Label: {rounded_true_label}, Predicted Score: {predicted_score}")
                print('------------------------')
        
        # calculate Spearman's ρ
        if len(set(predicted_scores)) == 1 or len(set(ground_truth_scores)) == 1:
            tqdm.write("Constant input detected in scores. Spearman's ρ is not defined for constant inputs.")
            spearman_rho = float('0.0')
        else:
            spearman_rho, _ = spearmanr(ground_truth_scores, predicted_scores)

        # print('cleaning model out from gpu...')
        # clean_model_out(model)
        
        return spearman_rho