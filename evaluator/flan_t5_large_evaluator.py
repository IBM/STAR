from tqdm import tqdm
from datasets import load_dataset, Dataset
from collections import Counter
import random
import torch
import re
from scipy.stats import spearmanr
from task_vector.util import clean_model_out


def balanced_sample(dataset, label_column, num_samples_per_label):
    label_counts = Counter(dataset[label_column])
    min_count = min(label_counts.values())
    
    num_samples_per_label = min(num_samples_per_label, min_count)
    
    sampled_data = []
    random.seed(42)
    for label in label_counts.keys():
        label_samples = [example for example in dataset if example[label_column] == label]
        sampled_data.extend(random.sample(label_samples, num_samples_per_label))

    return Dataset.from_dict({key: [sample[key] for sample in sampled_data] for key in dataset.column_names})

class Flan_t5_Large_evaluator:
    def __init__(self, dataset_name, cache_dir):
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        self.get_dataset_config()

    def get_dataset_config(self):
        if self.dataset_name == 'cola':
            ds = load_dataset("nyu-mll/glue", "cola", cache_dir=self.cache_dir)
            self.num_samples_per_label = 1000
            self.processed_ds = self.process_dataset(ds)
            self.max_new_tokens = 10
            self.generate_template_fn = self.generate_cola_template
            self.text_label_map = {
                '0': ['unacceptable', 'Unacceptable'],
                '1': ['acceptable', 'Acceptable'],                
            }
        elif self.dataset_name == 'qqp':
            ds = load_dataset("nyu-mll/glue", "qqp", cache_dir=self.cache_dir)
            self.num_samples_per_label = 1000
            self.processed_ds = self.process_dataset(ds)
            self.max_new_tokens = 10
            self.generate_template_fn = self.generate_qqp_template
            self.text_label_map = {
                '0': ['no', 'No'],
                '1': ['yes', 'Yes'],                
            }
        elif self.dataset_name == 'mnli':
            ds = load_dataset("nyu-mll/glue", "mnli_matched", cache_dir=self.cache_dir)
            self.num_samples_per_label = 400
            self.processed_ds = self.process_dataset(ds)
            self.max_new_tokens = 10
            self.generate_template_fn = self.generate_mnli_template
            self.text_label_map = {
                '0': ['entailment', 'Entailment'],
                '1': ['neutral', 'Neutral'],
                '2': ['contradiction', 'Contradiction', 'contradict', 'Contradict']
            }
        elif self.dataset_name == 'mrpc':
            ds = load_dataset("nyu-mll/glue", "mrpc", cache_dir=self.cache_dir)
            self.num_samples_per_label = 1000
            self.processed_ds = self.process_dataset(ds)
            self.max_new_tokens = 10
            self.generate_template_fn = self.generate_mrpc_template
            self.text_label_map = {
                '0': ['no', 'No'],
                '1': ['yes', 'Yes'],
            }
        elif self.dataset_name == 'qnli':
            ds = load_dataset("nyu-mll/glue", "qnli", cache_dir=self.cache_dir)
            self.num_samples_per_label = 1000
            self.processed_ds = self.process_dataset(ds)
            self.max_new_tokens = 10
            self.generate_template_fn = self.generate_qnli_template
            self.text_label_map = {
                '1': ['no', 'No'],
                '0': ['yes', 'Yes'],
            }
        elif self.dataset_name == 'rte':
            ds = load_dataset("nyu-mll/glue", "rte", cache_dir=self.cache_dir)
            self.num_samples_per_label = 1000
            self.processed_ds = self.process_dataset(ds)
            self.max_new_tokens = 10
            self.generate_template_fn = self.generate_rte_template
            self.text_label_map = {
                '1': ['no', 'No'],
                '0': ['yes', 'Yes'],
            }
        elif self.dataset_name == 'sst2':
            ds = load_dataset("nyu-mll/glue", "sst2", cache_dir=self.cache_dir)
            self.num_samples_per_label = 1000
            self.processed_ds = self.process_dataset(ds)
            self.max_new_tokens = 10
            self.generate_template_fn = self.generate_sst2_template
            self.text_label_map = {
                '1': ['positive', 'Positive'],
                '0': ['negative', 'Negative'],
            }
        elif self.dataset_name == 'stsb':
            ds = load_dataset("nyu-mll/glue", "stsb", cache_dir=self.cache_dir)
            self.processed_ds = self.process_dataset(ds)
            self.max_new_tokens = 10
            self.generate_template_fn = self.generate_stsb_template
        elif self.dataset_name == 'imdb':
            ds = load_dataset("stanfordnlp/imdb", cache_dir=self.cache_dir)
            self.num_samples_per_label = 1000
            self.processed_ds = self.process_dataset(ds)
            self.max_new_tokens = 10
            self.generate_template_fn = self.generate_imdb_template
            self.text_label_map = {
                '1': ['positive', 'Positive'],
                '0': ['negative', 'Negative'],
            }
        elif self.dataset_name == 'finance':
            ds = load_dataset("financial_phrasebank", "sentences_allagree", cache_dir=self.cache_dir)
            self.num_samples_per_label = 1000
            self.processed_ds = self.process_dataset(ds)
            self.max_new_tokens = 10
            self.generate_template_fn = self.generate_finance_template
            self.text_label_map = {
                '2': ['positive', 'Positive'],
                '1': ['neutral', 'Neutral'],
                '0': ['negative', 'Negative'],
            }
        elif self.dataset_name == 'agnews':
            ds = load_dataset("SetFit/ag_news", cache_dir=self.cache_dir)
            self.num_samples_per_label = 50
            self.processed_ds = self.process_dataset(ds)
            self.max_new_tokens = 10
            self.generate_template_fn = self.generate_agnews_template
            self.text_label_map = {
            '0': ['World', 'world'],
            '1': ['Sports', 'sports'],
            '2': ['Business', 'business'],
            '3': ['Sci/Tech', 'Sci/tech', 'sci/tech'],
        }
        elif self.dataset_name == 'hella':
            ds = load_dataset("Rowan/hellaswag", cache_dir=self.cache_dir)
            self.num_samples_per_label = 40
            self.processed_ds = self.process_dataset(ds)
            self.max_new_tokens = 10
            self.generate_template_fn = self.generate_hella_template
            self.text_label_map = {
            '0': ['0'],
            '1': ['1'],
            '2': ['2'],
            '3': ['3'],
        }
        elif self.dataset_name == 'boolq':
            ds = load_dataset("google/boolq", cache_dir=self.cache_dir)
            self.num_samples_per_label = 80
            self.processed_ds = self.process_dataset(ds)
            self.max_new_tokens = 10
            self.generate_template_fn = self.generate_boolq_template
            self.text_label_map = {
            '1': ['True', 'true'],
            '0': ['False', 'false'],
        }
        elif self.dataset_name == 'piqa':
            ds = load_dataset("ybisk/piqa", cache_dir=self.cache_dir)
            self.num_samples_per_label = 1000
            self.processed_ds = self.process_dataset(ds)
            self.max_new_tokens = 10
            self.generate_template_fn = self.generate_piqa_template
            self.text_label_map = {
            '1': ['2'],
            '0': ['1'],
        }
        else:
            raise ValueError(f"Dataset {self.dataset_name} is not supported.")

    def process_dataset(self, ds):
        print(f'Downloading datasets of {self.dataset_name}')

        if self.dataset_name in ['cola', 'sst2', 'stsb']:
            final_ds = ds['validation']
        elif self.dataset_name in ['qqp', 'mnli', 'rte', 'qnli', 'hella', 'piqa']:
            final_ds = balanced_sample(ds['validation'], label_column='label', num_samples_per_label=self.num_samples_per_label)
        elif self.dataset_name in ['boolq']:
            final_ds = balanced_sample(ds['validation'], label_column='answer', num_samples_per_label=self.num_samples_per_label)    
        elif self.dataset_name in ['mrpc', 'imdb', 'agnews']:
            final_ds = balanced_sample(ds['test'], label_column='label', num_samples_per_label=self.num_samples_per_label)

        elif self.dataset_name in ['finance']:
            split_ds = ds["train"].train_test_split(test_size=0.1, seed=42)
            final_ds = split_ds['test']   
        else:
            raise ValueError('Dataset name not matched')
        return final_ds

    
    def evaluate_stsb(self, model, tokenizer, device, print_output, batch_size):
        model.to(device)
        model.eval()
        sentence1s = [example['sentence1'] for example in self.processed_ds]
        sentence2s = [example['sentence2'] for example in self.processed_ds]
        true_labels = [example['label'] for example in self.processed_ds]  # Ground truth similarity scores
        
        predicted_scores = []
        ground_truth_scores = []
    
        ## For loop to get the LLM respond on input data
        for i in tqdm(range(0, len(true_labels), batch_size), desc=f"Evaluating model on {self.dataset_name}", leave=False):
            batch_sentence1s = sentence1s[i:i+batch_size]
            batch_sentence2s = sentence2s[i:i+batch_size]
            batch_labels = true_labels[i:i+batch_size]
            input_texts = [self.generate_template_fn(sentence1, sentence2) for sentence1, sentence2 in zip(batch_sentence1s, batch_sentence2s)]
            
            inputs = tokenizer.batch_encode_plus(input_texts, return_tensors='pt', padding=True, truncation=True)
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            
            # Generate output
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_new_tokens
                )
    
            generated_texts = [tokenizer.decode(output, skip_special_tokens=True).strip() for output in outputs]
    
            for generated_text, true_label in zip(generated_texts, batch_labels):
                remaining_text = generated_text.strip()
                match = re.search(r"[-+]?\d*\.\d+|\d+", remaining_text)
                if match:
                    remaining_text = match.group()
                    try:
                        predicted_score = float(remaining_text)
                    except ValueError:
                        print(f"Unable to convert generated text to float: {remaining_text}")
                        predicted_score = 0.0  
                else:
                    print(f"No valid number found in generated text: {generated_text}")
                    predicted_score = 0.0
                
                predicted_scores.append(predicted_score)
                rounded_true_label = round(true_label, 1)
                ground_truth_scores.append(rounded_true_label)
                
                if print_output:
                    print('------------------------')
                    print(f"Generated Text: {remaining_text}")
                    print(f"True Label: {rounded_true_label}, Predicted Score: {predicted_score}")
                    print('------------------------')
    
        # Calculate Spearman’s ρ
        if len(set(predicted_scores)) == 1 or len(set(ground_truth_scores)) == 1:
            print("Constant input detected in scores. Spearman's ρ is not defined for constant inputs.")
            spearman_rho = float('0.0')
        else:
            spearman_rho, _ = spearmanr(ground_truth_scores, predicted_scores)
    
        print('cleaning model out from gpu...')
        clean_model_out(model)
        return spearman_rho

    def evaluate(self, model, tokenizer, device, print_output, batch_size):
        model.to(device)
        model.eval()

        correct_predictions = 0
        total_predictions = 0

        ## Handling input data
        if self.dataset_name in ['cola', 'sst2', 'finance']:
            sentences = [example['sentence'] for example in self.processed_ds]
            true_labels = [example['label'] for example in self.processed_ds]
        elif self.dataset_name in ['qqp']:
            question1s = [example['question1'] for example in self.processed_ds]
            question2s = [example['question2'] for example in self.processed_ds]
            true_labels = [example['label'] for example in self.processed_ds]
        elif self.dataset_name in ['mnli']:
            premises = [example['premise'] for example in self.processed_ds]
            hypothesises = [example['hypothesis'] for example in self.processed_ds]
            true_labels = [example['label'] for example in self.processed_ds]
        elif self.dataset_name in ['mrpc', 'rte']:
            sentence1s = [example['sentence1'] for example in self.processed_ds]
            sentence2s = [example['sentence2'] for example in self.processed_ds]
            true_labels = [example['label'] for example in self.processed_ds]
        elif self.dataset_name in ['qnli']:
            questions = [example['question'] for example in self.processed_ds]
            sentences = [example['sentence'] for example in self.processed_ds]
            true_labels = [example['label'] for example in self.processed_ds]
        elif self.dataset_name in ['imdb']:
            texts = [example['text'] for example in self.processed_ds]
            true_labels = [example['label'] for example in self.processed_ds]
        elif self.dataset_name in ['agnews']:
            texts = [example['text'] for example in self.processed_ds]
            true_labels = [example['label'] for example in self.processed_ds]
        elif self.dataset_name in ['hella']:
            ctxs = [example['ctx'] for example in self.processed_ds]
            endingss = [example['endings'] for example in self.processed_ds]
            true_labels = [example['label'] for example in self.processed_ds]
        elif self.dataset_name in ['boolq']:
            questions = [example['question'] for example in self.processed_ds]
            passages = [example['passage'] for example in self.processed_ds]
            true_labels = [str(example['answer']) for example in self.processed_ds]
        elif self.dataset_name in ['piqa']:
            goals = [example['goal'] for example in self.processed_ds]
            sol1s = [example['sol1'] for example in self.processed_ds]
            sol2s = [example['sol2'] for example in self.processed_ds]
            true_labels = [example['label'] for example in self.processed_ds]
        else:
            raise ValueError('Dataset name not matched')

        ## For loop to get the LLM respond on input data
        for i in tqdm(range(0, len(true_labels), batch_size), desc=f"Evaluating model on {self.dataset_name}"):
            if self.dataset_name in ['cola', 'sst2', 'finance']:
                batch_sentences = sentences[i:i + batch_size]
                batch_labels = true_labels[i:i + batch_size]
                input_texts = [self.generate_template_fn(sentence) for sentence in batch_sentences]
            elif self.dataset_name in ['qqp']:
                batch_question1s = question1s[i:i+batch_size]
                batch_question2s = question2s[i:i+batch_size]
                batch_labels = true_labels[i:i+batch_size]
                input_texts = [self.generate_template_fn(question1, question2) for question1, question2 in zip(batch_question1s, batch_question2s)]
            elif self.dataset_name in ['mnli']:
                batch_premises = premises[i:i+batch_size]
                batch_hypothesises = hypothesises[i:i+batch_size]
                batch_labels = true_labels[i:i+batch_size]
                input_texts = [self.generate_template_fn(premise, hypothesis) for premise, hypothesis in zip(batch_premises, batch_hypothesises)]
            elif self.dataset_name in ['mrpc', 'rte']:
                batch_sentence1s = sentence1s[i:i+batch_size]
                batch_sentence2s = sentence2s[i:i+batch_size]
                batch_labels = true_labels[i:i+batch_size]
                input_texts = [self.generate_template_fn(sentence1, sentence2) for sentence1, sentence2 in zip(batch_sentence1s, batch_sentence2s)]
            elif self.dataset_name in ['qnli']:
                batch_questions = questions[i:i+batch_size]
                batch_sentences = sentences[i:i+batch_size]
                batch_labels = true_labels[i:i+batch_size]
                input_texts = [self.generate_template_fn(sentence, question) for sentence, question in zip(batch_sentences, batch_questions)]
            elif self.dataset_name in ['imdb']:
                batch_texts = texts[i:i+batch_size]
                batch_labels = true_labels[i:i+batch_size]
                input_texts = [self.generate_template_fn(text) for text in batch_texts]
            elif self.dataset_name in ['agnews']:
                batch_texts = texts[i:i+batch_size]
                batch_labels = true_labels[i:i+batch_size]
                input_texts = [self.generate_template_fn(text) for text in batch_texts]
            elif self.dataset_name in ['hella']:
                batch_ctxs = ctxs[i:i+batch_size]
                batch_endingss = endingss[i:i+batch_size]
                batch_labels = true_labels[i:i+batch_size]
                input_texts = [self.generate_template_fn(ctx, endings) for ctx ,endings in zip(batch_ctxs, batch_endingss)]
            elif self.dataset_name in ['boolq']:
                batch_questions = questions[i:i+batch_size]
                batch_passages = passages[i:i+batch_size]
                batch_labels = true_labels[i:i+batch_size]
                input_texts = [self.generate_template_fn(question, passage) for question, passage in zip(batch_questions, batch_passages)]
            elif self.dataset_name in ['piqa']:
                batch_goals = goals[i:i+batch_size]
                batch_sol1s = sol1s[i:i+batch_size]
                batch_sol2s = sol2s[i:i+batch_size]
                batch_labels = true_labels[i:i+batch_size]
                input_texts = [self.generate_template_fn(goal,sol1, sol2) for goal,sol1, sol2 in zip(batch_goals, batch_sol1s, batch_sol2s)]
            
            inputs = tokenizer.batch_encode_plus(input_texts, return_tensors='pt', padding=True, truncation=True)
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)

            # Generate output
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_new_tokens
                )

            generated_texts = [tokenizer.decode(output, skip_special_tokens=True).strip() for output in outputs]

            for generated_text, true_label in zip(generated_texts, batch_labels):
                remaining_text = generated_text.strip()
                predicted_label = -1
                hit = False

                if remaining_text:
                    for label_index, labels in self.text_label_map.items():
                        for label in labels:
                            if label in remaining_text:
                                predicted_label = int(label_index)
                                hit = True
                                break
                        if hit:
                            break
                    if not hit:
                        tqdm.write(f"<Unexpected model outputs: {remaining_text}>") 
                else:
                    # generate nothing
                    tqdm.write(f"<Nothing is generating>") 
                    predicted_label = -1
                
                if self.dataset_name in ['hella', 'boolq']:
                    for possible_label, possible_text in self.text_label_map.items():
                        if true_labels[i] in possible_text:
                            true_label = int(possible_label)
                            break        
                if print_output:
                    print('------------------------')
                    print(f"Generated Text: {remaining_text}")
                    print(f"True Label: {true_label}, Predicted Label {predicted_label}")
                    print('------------------------')

                if predicted_label == true_label:
                    correct_predictions += 1

                total_predictions += 1

        accuracy = correct_predictions / total_predictions
        print('cleaning model out from gpu...')
        clean_model_out(model)
        return accuracy
    
    def generate_cola_template(self, sentence):
        return f"Indicate if the following sentence is grammatically correct or not: \"{sentence}\". Answer ‘acceptable’ or ‘unacceptable’."
    def generate_qqp_template(self, question1, question2):
        return f"Do the questions ‘{question1}’ and ‘{question2}’ have the same intent? Answer with ‘yes’ or ‘no’."
    def generate_mnli_template(self, premise, hypothesis):
        return f"Does the premise: ‘{premise}’ logically imply, contradict, or is neutral to the hypothesis: ‘{hypothesis}’? Answer with ‘entailment’, ‘contradiction’, or ‘neutral’."
    def generate_mrpc_template(self, sentence1, sentence2):
        return f"Are the following sentences ‘{sentence1}’ and ‘{sentence2}’ conveying the same meaning? Answer with ‘yes’ or ‘no’."
    def generate_qnli_template(self, sentence, question):
        return f"Given the context: ‘{sentence}’, does the question ‘{question}’ have an answer based on the information provided? Answer with ‘yes’ or ‘no’."
    def generate_rte_template(self, sentence1, sentence2):
        return f"Does the text: ‘{sentence1}’ entail that ‘{sentence2}’ is true? Provide ‘yes’ or ‘no’."
    def generate_sst2_template(self, sentence):
        return f"Given the sentence ‘{sentence}’, determine the sentiment. Is it positive or negative?"
    def generate_stsb_template(self, sentence1, sentence2):
        return f"Consider the sentences ‘{sentence1}’ and ‘{sentence2}’. On a scale from 1 (completely different) to 5 (completely similar), rate the similarity."
    def generate_imdb_template(self, review_text):
        prompt_prefix = "The following is a movie review: '"
        prompt_suffix = "' Please classify its sentiment as either 'positive' or 'negative'."
        return f"{prompt_prefix}{review_text}{prompt_suffix}"
    
    def generate_finance_template(self, sentence):
        return f"The following is a sentence from a financial report: '{sentence}'. Please classify its sentiment as either 'positive', 'negative', or 'neutral'."
    def generate_agnews_template(self, article_text):
        prompt_prefix = "The following is a news article: '"
        prompt_suffix = "' Please classify it into one of the following categories: World, Sports, Business, or Sci/Tech."
        return f"{prompt_prefix}{article_text}{prompt_suffix}"
    def generate_hella_template(self, context, endings):
        prompt_prefix = "The following is a context: '"
        prompt_suffix = "'\nGiven this context, choose the most appropriate ending:\n"
        options = "\n".join([f"{i}: {ending}" for i, ending in enumerate(endings)])
        prompt_end = "Please return 0, 1, 2, or 3 as the correct answer."
        return f"{prompt_prefix}{context}{prompt_suffix}{options}\n{prompt_end}"
    def generate_boolq_template(self, question, passage):
        return f"Based on the following passage: ‘{passage}’, answer the question: ‘{question}’. Respond with ‘true’ or ‘false’."
    def generate_piqa_template(self, goal, sol1, sol2):
        return f"""Question: {goal}
    Option 1: {sol1}
    Option 2: {sol2}
    Which option is more appropriate? Respond with 'Option 1' or 'Option 2'."""
