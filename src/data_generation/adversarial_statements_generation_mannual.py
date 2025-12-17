from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, pipeline
import json
from pydantic import BaseModel
from typing import Literal
import random
import re

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data
    
class Sample():
    def __init__(self, statements, labels, task_instructions):
        # 1&1, 0&0, 1&0, 0&1, 1||1, 0||0, 1||0, 0||1
        self.statements = statements
        self.labels = labels
        self.task_instructions = task_instructions
        self.true_nums = sum(self.labels)
    def sample(self):
        if(self.true_nums < 2 or len(self.statements) - self.true_nums < 2):
            return False
        true_statements = self.statements[:self.true_nums]
        false_statements = self.statements[self.true_nums:]
        true_idx = random.choices(range(len(true_statements)), k=2)
        false_idx = random.choices(range(len(false_statements)), k=2)
        # sample 4 statements
        for _ in range(2):
            true_idx.append(random.randint(0, len(true_statements) - 1))
            false_idx.append(random.randint(0, len(false_statements) - 1))
        
        self.sampled_true_statements = [true_statements[i] for i in true_idx]
        self.sampled_false_statements = [false_statements[i] for i in false_idx]
        self.sampled_labels = [self.labels[i] for i in true_idx] + [self.labels[i + self.true_nums] for i in false_idx]
        return True
    def samples_statement_a_b(self):
        """
        Args:
            samples (dict): true_statements, false_statements, true_labels, false_labels
            task_instructions (list): list of task instructions
        Returns:
            input_dicts (list): list of input dicts for prompt
        """
        
        input_dicts = []
        sign_label = [(1,1), (0,0), (1,0), (0,1)]
        idx_list = [(0,1), (2,3), (0,2), (1,3)]
        for i in range(8):
            sign_a, sign_b = sign_label[i%4]
            idx_a, idx_b = idx_list[i%4]
            input_dicts.append({
                "original_statement_a": self.sampled_true_statements[idx_a] if sign_a == 1 else self.sampled_false_statements[idx_a],
                "original_statement_b": self.sampled_true_statements[idx_b] if sign_b == 1 else self.sampled_false_statements[idx_b]
            })
        
        return input_dicts

    def samples_labels(self):
        labels = []
        # 1&1, 1&0, 0&1, 0&0, 1||1, 0||0, 0||1, 1||0
        sign_label = [(1,1), (0,0), (1,0), (0,1)]
        idx_list = [(0,1), (2,3), (0,2), (1,3)]
        for i in range(8):
            sign_a, sign_b = sign_label[i%4]
            idx_a, idx_b = idx_list[i%4]
            label_a = self.sampled_labels[idx_a] if sign_a == 1 else self.sampled_labels[idx_a + 4]
            label_b = self.sampled_labels[idx_b] if sign_b == 1 else self.sampled_labels[idx_b + 4]
            labels.append((label_a and label_b) if i < 4 else (label_a or label_b))
        
        return labels

if __name__ == "__main__":
    data = load_json('./data/test_examples_with_csv.json')
    pbar = tqdm(enumerate(data.items()), total=len(data))
    
    new_data = {}
    tot = 0
    tot_table = 0
    generated_true = 0
    tasks_instructions = ["AND", "OR"]
    task_list = [tasks_instructions[0] for _ in range(4)] + [tasks_instructions[1] for _ in range(4)]
    
    for i, (key, value) in pbar:
        original_statements = value[0]
        original_labels = value[1]
        extend_statements = []
        
        sample = Sample(original_statements, original_labels, task_list)
        if not sample.sample():
            continue
        tot_table += 1
        # 1&1, 1&0, 0&1, 0&0, 1||1, 0||0, 1||0, 0||1
        input_dicts = sample.samples_statement_a_b()
        extend_labels = sample.samples_labels()
        
        # count labels balance
        generated_true += sum(extend_labels)
        tot += len(extend_labels)
        
        pbar.set_description(f"Generated true statements ratio: {generated_true}/{tot}")
        
        for input_dict, task_instruction in zip(input_dicts, task_list):
            new_statement = input_dict["original_statement_a"] + " " + task_instruction + " " + input_dict["original_statement_b"]
            extend_statements.append(new_statement)
            
        new_data[key] = [extend_statements, extend_labels, value[2], value[3]]
        
    with open('./data/test_examples_with_csv_adversarial.json', 'w') as f:
        json.dump(new_data, f, indent=4)
    print("tot_table:", tot_table)