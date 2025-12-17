from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, pipeline
from torch.utils.data import Dataset, DataLoader

from pydantic import BaseModel
from typing import Literal
from tqdm import tqdm
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
    parser.add_argument('--data_path', type=str, default='./data/promptDataset/single_prompted_test_examples.json', help='Path to the input JSON data file')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-7B-Instruct-1M', help='Name of the pre-trained model to use')
    parser.add_argument('--max_new_tokens', type=int, default=10, help='Maximum number of new tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.0, help='Temperature for text generation')
    args = parser.parse_args()
    
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = AutoModelForCausalLM.from_pretrained(model_name)
    hf_pipe = pipeline(
        "text-generation",
        model=llm,
        tokenizer=tokenizer,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        return_full_text=False,
    )
    hf_llm = HuggingFacePipeline(pipeline=hf_pipe)
    
    data = load_json(args.data_path)
    dataset = PromptDataset(data)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    pbar = tqdm(dataloader, total=len(dataloader))
    
    parser = JsonOutputParser(pydantic_object=ResponseSchema)
    
    correct = 0
    wrong = 0
    total = 0
    for prompts, labels in pbar:
        result = hf_llm.generate(prompts)
        for i in range(len(result.generations)):
            generation = result.generations[i][0].text
            parsed = parser.parse(generation)
            if parsed.answer == labels[i]:
                correct += 1
            else:
                wrong += 1
            total += 1
        pbar.set_description(f"Accuracy: {correct}/{total} = {correct/total:.4f}")
        