from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, pipeline
import json
from pydantic import BaseModel
from typing import Literal
import re

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def extract_json(text):
    match = re.search(r'\{.*\}', text, re.S)
    return match.group(0) if match else None
    

class StatementSchema(BaseModel):
    new_statement: str

if __name__ == "__main__":
    model_name = "Qwen/Qwen2.5-7B-Instruct-1M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = AutoModelForCausalLM.from_pretrained(model_name)
    hf_pipe = pipeline(
        "text-generation",
        model=llm,
        tokenizer=tokenizer,
        temperature=0.0,
        max_new_tokens=128,
        do_sample=False,
        return_full_text=False,
    )
    hf_llm = HuggingFacePipeline(pipeline=hf_pipe)
    
    data = load_json('./data/test_examples_with_csv.json')
    pbar = tqdm(enumerate(data.items()))
    
    parser = JsonOutputParser(pydantic_object=StatementSchema)
    
    template = """You are given a statement. Your task is to generate one paraphrased statement that preserves the exact same meaning and truth conditions as the original, while using clearly different wording. The paraphrased statement should be semantically equivalent to the original.
    
    {format_instructions}
    
    Original Statement: {original_statement}
    """
    
    prompt = PromptTemplate(
        template = template,
        input_variables = ["original_statement"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    chain = prompt | hf_llm
    new_data = {}
    
    for i, (key, value) in pbar:
        original_statements = value[0]
        original_labels = value[1]
        extend_statements = []
        extend_labels = []
        for statement, label in zip(original_statements, original_labels):
            result = chain.invoke({"original_statement": statement})
            output_text = extract_json(result)
            if output_text is None:
                output_text = result
                print("Parsing error for statement:", statement)
                print("Parsing error output:", result)
            else:
                try:   
                    output_text = parser.parse(output_text)
                    output_text = output_text['new_statement']
                except:
                    output_text = result
                
            extend_statements.append(output_text)
            extend_labels.append(label)
            
        new_data[key] = [extend_statements, extend_labels, value[2], value[3]]
        
    with open('./data/test_examples_with_csv_paraphrased.json', 'w') as f:
        json.dump(new_data, f, indent=4)
    