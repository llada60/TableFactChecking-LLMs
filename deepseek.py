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
    model_name = "deepseek-ai/DeepSeek-V2-Lite"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
    model.generation_config = GenerationConfig.from_pretrained(model_name)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id
    
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
    for key, value in tqdm(json.items(), desc=f"acc:{correct}/{total}"):
        statements = value[0]
        labels = value[1]
        table_title =value[2]
        table = value[3]
        total += len(statements)
        for statement, label in zip(statements, labels):
            text = prompt.format(statement=statement, table=table, table_title=table_title)
            inputs = tokenizer(text, return_tensors='pt').to(model.device)
            input_size = inputs['input_ids'].shape[-1]
            outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)
            output_text = tokenizer.decode(outputs[0][input_size:], skip_special_tokens=True)
            if "supported" in output_text.lower():
                pred_label = True
            elif "refuted" in output_text.lower():
                pred_label = False
            else:
                pred_label = None
            if pred_label == label:
                correct += 1
            elif pred_label is not None:
                wrong += 1
                
        print(f"correct: {correct}, wrong: {wrong}, total: {total}, accuracy: {correct/total:.4f}")
    