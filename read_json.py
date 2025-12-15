import json
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, pipeline

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

datas = load_json('./data/test_examples_with_csv.json')
max_len = 0
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct-1M")
llm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct-1M")
print(llm.config.max_position_embeddings)

for key, value in datas.items():
    table = value[3]
    tokens = tokenizer(table)
    max_len = max(max_len, len(tokens['input_ids']))
    
print("Max table length:", max_len)