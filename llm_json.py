from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, pipeline

from pydantic import BaseModel
from typing import Literal
from tqdm import tqdm
import json
import argparse


def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

class ResponseSchema(BaseModel):
    # reasoning: str
    answer: Literal["Supported", "Refuted"]
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/test_examples_with_csv.json', help='Path to the input JSON data file')
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
    
    datas = load_json(args.data_path)
    pbar = tqdm(datas.items())
    
    parser = JsonOutputParser(pydantic_object=ResponseSchema)
    
    # Analyze in one or two sentences and 
    template = """
    You will be provided with a statement and a table. Determine whether the statement is supported or refuted by the information in the table.
    
    {format_instructions}
    
    Statement: {statement}
    
    Table Title: {table_title}
    
    Table: {table}
    
    """
    prompt = PromptTemplate(
        template = template,
        input_variables = ["statement", "table_title", "table"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    chain = prompt | hf_llm
    
    correct = 0
    wrong = 0
    total = 0
    for key, value in pbar:
        statements = value[0]
        labels = value[1]
        table_title =value[2]
        table = value[3]
        pbar.set_description(f"acc:{correct}/{total}, wrong:{wrong}")
        total += len(statements)
        for statement, label in zip(statements, labels):
            result = chain.invoke({
                "statement": statement,
                "table_title": table_title,
                "table": table
            })
            try:
                output_text = parser.parse(result)
                output_text = output_text['answer']
            except:
                output_text = result
                print("table: ", table)
                print("Parsing error for statement:", statement)
                
            pred = None
            if(output_text is not None):
                if "support" in output_text.lower():
                    pred = True
                elif "refute" in output_text.lower():
                    pred = False
            else: continue
            if pred == label:
                correct += 1
            else:
                wrong += 1
    # print(f"Final Accuracy: {correct}/{total}, wrong:{wrong}")