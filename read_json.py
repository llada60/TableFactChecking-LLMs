from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, pipeline
from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser
from langchain_core.prompts import PromptTemplate

from pydantic import BaseModel
from typing import Literal
import json

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
    # transfer to flattened json
    # flatten_data = json2dicList('./data/test_examples_with_csv.json') 
    # with open('./data/flattened_test_examples_with_csv.json', 'w') as f:
    #     json.dump(flatten_data, f)
        
    json_path = './data/flattened_test_examples_with_csv.json'
    with open(json_path, 'r') as f:
        data = json.load(f)
        
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
    
    promptDataset = dicList2prompt(data, single_prompt, parser)
    with open('./data/promptDataset/single_prompted_test_examples.json', 'w') as f:
        json.dump(promptDataset, f)


