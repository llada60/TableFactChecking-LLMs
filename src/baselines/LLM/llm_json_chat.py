from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, pipeline

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

def build_message(statement, table_title, table):
    messages = [
        {
            "role": "system",
            "content": """
                You will be provided with a statement and a table. Determine whether the statement is supported or refuted by the information in the table.
                {format_instructions}""".format(format_instructions=parser.get_format_instructions())
        },
        {
            "role": "assistant",
            "content": """Please provide me with the statement and the table you are referring to."""
        },
        {
            "role": "user",
            "content": f"""
                The table title is {table_title}
                The table is {table}"""
        },
        {
            "role": "assistant",
            "content": """Please provide me with the statement you would like to verify using the provided table."""
        },
        {
            "role": "user",
            "content": f"""The statement is {statement}. Please determine whether the statement is Supported or Refuted by the table."""
        }
    ]
    return messages

class ResponseSchema(BaseModel):
    # reasoning: str
    answer: Literal["Supported", "Refuted"]
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/test_examples_with_csv_paraphrased.json', help='Path to the input JSON data file')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-7B-Instruct-1M', help='Name of the pre-trained model to use')
    parser.add_argument('--max_new_tokens', type=int, default=20, help='Maximum number of new tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.0, help='Temperature for text generation')
    parser.add_argument('--do_sample', type=bool, default=False, help='Whether to use sampling for text generation')
    args = parser.parse_args()
    print(args)
    
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    llm = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map='auto')
    llm.eval()
    
    data = load_json(args.data_path)
    pbar = tqdm(data, total=len(data))
    
    parser = JsonOutputParser(pydantic_object=ResponseSchema)
    
    correct = 0
    wrong = 0
    total = 0
    for value in pbar:
        total += 1
        pbar.set_description(f"acc:{correct}/{total}, wrong:{wrong}")
        
        label = value["label"]
        message_text = build_message(
            statement=value["statement"],
            table_title=value["table_title"],
            table=value["table"]
        )
        prompt = tokenizer.apply_chat_template(
            message_text,
            tokenize=False,
            add_generation_prompt=True
        )
        tokens = tokenizer(prompt, return_tensors='pt').to(llm.device)
        outputs = llm.generate(
            **tokens,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,   
        )[0][tokens['input_ids'].shape[-1]:]
        result = tokenizer.decode(outputs, skip_special_tokens=True)
        # pbar.set_description(f"{result}")
        # print(result, flush=True)
        
        json_str = extract_json(result)
        if json_str is None:
            json_str = result
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

    print(f"Final Accuracy: {correct}/{total}, wrong:{wrong}")