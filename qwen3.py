import json
import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import outlines
from pydantic import BaseModel, Field
from typing import Literal

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

class ResponseWithoutReasoning(BaseModel):
    answer: Literal["Supported", "Refuted"]

class ResponseWithReasoning(BaseModel):
    reasoning: str
    answer: Literal["Supported", "Refuted"]

if __name__ == "__main__":
    model_name = "Qwen/Qwen2.5-7B-Instruct-1M"
    # llm_name = "Qwen/Qwen3-1.7B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  
        device_map="cuda"           
    )
    model = outlines.from_transformers(
        llm,
        tokenizer
    )
    
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
            outputs = model.generate(text, schema=ResponseWithoutReasoning, max_new_tokens=250)
            # it will output 1. supported. 2. refuted. Answer: ... (hard to directly parse)
            print(outputs)
            if "support" in outputs['answer'].lower():
                pred_label = True
            elif "refute" in outputs['answer'].lower():
                pred_label = False
            else:
                pred_label = None
                
            if pred_label == label:
                correct += 1
            elif pred_label is not None:
                wrong += 1
                
        print(f"correct: {correct}, wrong: {wrong}, total: {total}, accuracy: {correct/total:.4f}")
    