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

def extract_json(text):
    match = re.search(r'\{.*\}', text, re.S)
    return match.group(0) if match else None

tasks_instructions = [
    "Your task is to generate one single statement that combines statements A and B using logical AND, such that the resulting statement is true if and only if both A and B are true.",
    "Your task is to generate one single statement that combines statements A and B using logical OR, such that the resulting statement is true if and only if at least one of A or B is true."]

class StatementSchema(BaseModel):
    new_statement: str
    
def samples_statement_a_b(samples, task_instructions):
    """
    Args:
        samples (dict): true_statements, false_statements, true_labels, false_labels
        task_instructions (list): list of task instructions
    Returns:
        input_dicts (list): list of input dicts for prompt
    """
    # 1&1, 1&0, 0&1, 0&0, 1||1, 0||0, 0||1, 1||0, 
    input_dicts = []
    for i in range(8):
        input_dicts.append({
            "task": task_instructions[i],
            "original_statement_a": samples["true_statements"][i] if i in [0, 1, 4, 7] else samples["false_statements"][i],
            "original_statement_b": samples["true_statements"][i] if i%2 == 0 else samples["false_statements"][i],
        })
    
    return input_dicts

def samples_labels(samples):
    labels = []
    for i in range(8):
        label_a = samples["true_statements"][i] if i in [0, 1, 4, 7] else samples["false_statements"][i]
        label_b = samples["true_statements"][i] if i%2 == 0 else samples["false_statements"][i]
        labels.append((label_a and label_b) if i < 4 else (label_a or label_b))
    
    return labels

if __name__ == "__main__":
    model_name = "Qwen/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = AutoModelForCausalLM.from_pretrained(model_name)
    hf_pipe = pipeline(
        "text-generation",
        model=llm,
        tokenizer=tokenizer,
        temperature=0.0,
        max_new_tokens=1024,
        do_sample=False,
        return_full_text=False,
    )
    hf_llm = HuggingFacePipeline(pipeline=hf_pipe)
    
    data = load_json('./data/test_examples_with_csv.json')
    pbar = tqdm(enumerate(data.items()))
    
    parser = JsonOutputParser(pydantic_object=StatementSchema)
    
    template = """You are given two statements A and B. {task} Do not change, paraphrase, or replace any words used in the original statements A and B.
    
    {format_instructions}
    
    Statement A: {original_statement_a}
    Statement B: {original_statement_b}
    """
    
    prompt = PromptTemplate(
        template = template,
        input_variables = ["task", "original_statement_a", "original_statement_b"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    chain = prompt | hf_llm
    new_data = {}
    
    for i, (key, value) in pbar:
        original_statements = value[0]
        original_labels = value[1]
        extend_statements = []
        
        true_statements = original_statements[:len(original_statements)//2]
        false_statements = original_statements[len(original_statements)//2:]
        # sample 8 statements
        true_idx = [random.randint(0, len(true_statements)-1) for _ in range(8)]
        false_idx = [random.randint(0, len(false_statements)-1) for _ in range(8)]
        samples = {
            "true_statements": [true_statements[i] for i in true_idx],
            "false_statements": [false_statements[i] for i in false_idx],
            "true_labels": [original_labels[i] for i in true_idx],
            "false_labels": [original_labels[i] for i in false_idx],
        }
        task_list = tasks_instructions[0] * 4 + tasks_instructions[1] * 4
        input_dicts = samples_statement_a_b(samples, task_list)
        extend_labels = samples_labels(samples)
        
        # 1&1, 1&0, 0&1, 0&0, 1||1, 0||0, 1||0, 0||1
        for input_dict in input_dicts:
            result = chain.invoke(input_dict)
            output_text = extract_json(result)
            if output_text is None:
                output_text = result
                print("Parsing error for statements:", input_dict["original_statement_a"], input_dict["original_statement_b"])
                print("Parsing error output:", result)
            else:
                try:   
                    output_text = parser.parse(output_text)
                    output_text = output_text['new_statement']
                except:
                    output_text = result
            extend_statements.append(output_text)
            
        new_data[key] = [extend_statements, extend_labels, value[2], value[3]]
        
    with open('./data/test_examples_with_csv_adversarial.json', 'w') as f:
        json.dump(new_data, f)
    