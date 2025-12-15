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

class StatementSchema(BaseModel):
    statement1: str
    statement2: str
    statement3: str
    statement4: str
    statement5: str

if __name__ == "__main__":
    model_name = "Qwen/Qwen2.5-7B-Instruct-1M"
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
    
    template = """
    You are to generate five different statements which have the same meaning as the given statement. Ensure that each statement is unique in its wording while conveying the same information.
    
    {format_instructions}
    
    Original Statement: {original_statement}
    """
    
    prompt = PromptTemplate(
        template = template,
        input_variables = ["original_statement"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    chain = prompt | hf_llm | parser
    
    for i, (key, value) in pbar:
        original_statements = value[0]
        original_labels = value[1]
        extend_statements = []
        extend_labels = []
        for statement, label in zip(original_statements, original_labels):
            try:
                result = chain.invoke({"original_statement": statement})
            except:
                print("Error processing statement:", i)
                continue
            extend_statements += [
                statement,
                result['statement1'],
                result['statement2'],
                result['statement3'],
                result['statement4'],
                result['statement5']
            ]
            extend_labels += label * 6
            
        datas[key][0] = extend_statements
        datas[key][1] = extend_labels
    