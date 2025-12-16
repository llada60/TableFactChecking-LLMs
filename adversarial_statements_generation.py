from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, pipeline
import json
from pydantic import BaseModel
from typing import Literal

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

tasks_instructions = [
    "Your task is to generate one single statement that combines statements A and B using logical AND, such that the resulting statement is true if and only if both A and B are true.",
    "Your task is to generate one single statement that combines statements A and B using logical OR, such that the resulting statement is true if and only if at least one of A or B is true."]

class StatementSchema(BaseModel):
    new_statement: str

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
    
    datas = load_json('./data/test_examples_with_csv.json')
    pbar = tqdm(enumerate(datas.items()))
    
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
    
    chain = prompt | hf_llm | parser
    new_data = {}
    
    for i, (key, value) in pbar:
        original_statements = value[0]
        original_labels = value[1]
        extend_statements = []
        extend_labels = []
        true_statements = original_statements[:len(original_statements)//2]
        false_statements = original_statements[len(original_statements)//2:]
        
        # 1&1, 1&0, 0&1, 0&0, 1||1, 0||0, 1||0, 0||1
        task_list = tasks_instructions[0] * 4 + tasks_instructions[1] * 4
        
            
        new_data[key] = [extend_statements, extend_labels, value[2], value[3]]
        
    with open('./data/test_examples_with_csv_adversarial.json', 'w') as f:
        json.dump(new_data, f)
    