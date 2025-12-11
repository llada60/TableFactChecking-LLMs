import json
import os
from openai import OpenAI
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    json = load_json('./data/test_examples_with_csv.json')
    prompt = """
    Statement: {statement}
    Table Title: {table_title}
    Table: {table}
    Fact-verification: Based on the information provided in the table, is the statement supported or refuted?
    """
    correct = 0
    wrong = 0
    total = 0
    pbar = tqdm(json.items())
    for key, value in pbar:
        statements = value[0]
        labels = value[1]
        table_title =value[2]
        table = value[3]
        total += len(statements)
        pbar.set_description(f"acc:{correct}/{total}")
        
        for statement, label in zip(statements, labels):
            text = prompt.format(statement=statement, table=table, table_title=table_title)
            inputs = tokenizer(text, return_tensors='pt').to(model.device)
            input_size = inputs['input_ids'].shape[-1]
            outputs = model.generate(**inputs, max_new_tokens=100)
            output_text = tokenizer.decode(outputs[0][input_size:], skip_special_tokens=True)
            # it will output 1. supported. 2. refuted. Answer: ... (hard to directly parse)
            print(output_text)
            if "support" in output_text.lower():
                pred_label = True
            elif "refute" in output_text.lower():
                pred_label = False
            else:
                pred_label = None
                
            if pred_label == label:
                correct += 1
            elif pred_label is not None:
                wrong += 1
                
        print(f"correct: {correct}, wrong: {wrong}, total: {total}, accuracy: {correct/total:.4f}")
    