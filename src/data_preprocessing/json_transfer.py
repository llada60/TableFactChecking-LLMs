from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, pipeline
from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser
from langchain_core.prompts import PromptTemplate

from pydantic import BaseModel
from typing import Literal
import json
import argparse
import os

def json2dicList(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    flatten_data = []
    idx = 0
    for key, value in data.items():
        statements = value[0]
        labels = value[1]
        table_title = value[2]
        table = value[3]
        for statement, label in zip(statements, labels):
            flatten_data.append({
                "id": idx,
                "statement": statement,
                "label": label,
                "table_title": table_title,
                "table": table
            })
            idx += 1
    return flatten_data

def dicList2prompt(dic_list, prompt_template, parser):
    promptDataset = []
    for item in dic_list:
        prompt = prompt_template.format(
            statement=item['statement'],
            table_title=item['table_title'],
            table=item['table']
        )
        promptDataset.append(
            {
                "id": item['id'],
                "prompt": prompt,
                "label": item['label']
            }
        )
    return promptDataset
    

class ResponseSchema(BaseModel):
    # reasoning: str
    answer: Literal["Supported", "Refuted"]
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../../data/test_examples_with_csv_paraphrased.json', help='Path to the input JSON data file')
    parser.add_argument('--prompt_type', type=str, default='direct_prompt', help='Type of prompt to generate: single or multi')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-7B-Instruct-1M', help='Name of the pre-trained model to use')
    parser.add_argument('--max_new_tokens', type=int, default=10, help='Maximum number of new tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.0, help='Temperature for text generation')
    args = parser.parse_args()
    
    # transfer to flattened json
    flatten_data = json2dicList(args.data_path) 
    flatten_data_path = args.data_path.replace('.json', '_flattened.json')
    with open(flatten_data_path, 'w') as f:
        json.dump(flatten_data, f)
        
    # with open(flatten_data_path, 'r') as f:
    #     flattened_data = json.load(f)
        
    # save prompt label json
    parser = JsonOutputParser(pydantic_object=ResponseSchema)
        
        # Analyze in one or two sentences and 
    single_prompt_template = """You will be provided with a statement and a table. Determine whether the statement is supported or refuted by the information in the table.
    {format_instructions}
    
    Statement: {statement}
    Table Title: {table_title}
    Table: {table}
    """
    single_prompt = PromptTemplate(
        template = single_prompt_template,
        input_variables = ["statement", "table_title", "table"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    promptDataset = dicList2prompt(flatten_data, single_prompt, parser)
    file_name = os.path.basename(args.data_path)
    base_path = os.path.join(os.path.dirname(args.data_path), "promptDataset")
    with open(os.path.join(base_path, file_name.replace('.json', f'_{args.prompt_type}.json')), 'w') as f:
        json.dump(promptDataset, f)


