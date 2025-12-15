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

class ResponseSchema(BaseModel):
    # reasoning: str
    answer: Literal["Supported", "Refuted"]
    

if __name__ == "__main__":
    model_name = "Qwen/Qwen2.5-7B-Instruct-1M"
    # model_name = "Qwen/Qwen3-1.7B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = AutoModelForCausalLM.from_pretrained(model_name)
    hf_pipe = pipeline(
        "text-generation",
        model=llm,
        tokenizer=tokenizer,
        temperature=0.0,
        max_new_tokens=16,
        do_sample=False,
        return_full_text=False,
    )
    hf_llm = HuggingFacePipeline(pipeline=hf_pipe)
    
    datas = load_json('./data/test_examples_with_csv.json')
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
    chain = prompt | hf_llm | parser
    
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
            output_text = chain.invoke({
                "statement": statement,
                "table_title": table_title,
                "table": table
            })
            if(output_text is not None):
                if "support" in output_text['answer'].lower():
                    pred = True
                elif "refute" in output_text['answer'].lower():
                    pred = False
            else:
                pred = None
                
            if pred == label:
                correct += 1
            else:
                wrong += 1
    print(f"Final Accuracy: {correct}/{total}, wrong:{wrong}")