from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser
from langchain_core.prompts import PromptTemplate

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, pipeline
from torch.utils.data import Dataset, DataLoader

from pydantic import BaseModel
from typing import Literal
from tqdm import tqdm
import torch
import json
import argparse
import re


def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def extract_json(text):
    match = re.search(r'\{.*\}', text, re.S)
    return match.group(0) if match else None

class ResponseSchema(BaseModel):
    # reasoning: str
    answer: Literal["Supported", "Refuted"]
    
class PromptDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.len = len(data)
    def __len__(self):
        return self.len
    def __getitem__(self, idx):
        item  = self.data[idx]
        return item["prompt"], item["label"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/promptDataset/test_examples_with_csv_direct_prompt.json', help='Path to the input JSON data file')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-7B-Instruct-1M', help='Name of the pre-trained model to use')
    parser.add_argument('--max_new_tokens', type=int, default=10, help='Maximum number of new tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.0, help='Temperature for text generation')
    args = parser.parse_args()
    
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', trust_remote_code=True)
    llm = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    llm.eval()
    
    data = load_json(args.data_path)
    dataset = PromptDataset(data)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    pbar = tqdm(dataloader, total=len(dataloader))
    
    parser = JsonOutputParser(pydantic_object=ResponseSchema)
    
    correct = 0
    wrong = 0
    total = len(dataset)
    for prompts, labels in pbar:
        prompts = list(prompts)
        tokens = tokenizer(prompts, return_tensors='pt', padding=True).to(llm.device)
        outputs = llm.generate(
            **tokens,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=False,
        )
        results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for i, (res, label) in enumerate(zip(results, labels)):
            res = res[len(prompts[i]):]  
            print(res)
            json_str = extract_json(res)
            if json_str is None:
                continue
            try:
                parsed_output = parser.parse(json_str)["answer"]
            except: 
                parsed_output = json_str
            
            if 'support' in parsed_output.lower():
                parsed_answer = True
            elif 'refute' in parsed_output.lower():
                parsed_answer = False
            else:
                continue

            if parsed_answer == label:
                correct += 1
            else:
                wrong += 1

        pbar.set_description(f"Accuracy: {correct}/{total}, Wrong: {wrong}")
    
    print(f"Final Accuracy: {correct}/{total}, Wrong: {wrong}")