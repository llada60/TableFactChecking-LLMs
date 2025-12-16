from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

if __name__ == "__main__":
    model_name = "google/gemma-3-1b-it"
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
    model = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it")
    correct = 0
    wrong = 0
    total = 0
    json = load_json('./data/test_examples_with_csv.json')
    
    prompt = """
    Statement: {statement}
    Table Title: {table_title}
    Table: {table}
    Fact-verification: Based on the information provided in the table, is the statement supported or refuted?
    """
    pbar = tqdm(json.items())
    for key, value in pbar:
        statements = value[0]
        labels = value[1]
        table_title =value[2]
        table = value[3]
        total += len(statements)
        pbar.set_description(f"acc:{correct}/{total}")
        
        for statement, label in zip(statements, labels):
            text = prompt.format(statement=statement, table=table, table_title=table_title)
            inputs = tokenizer(text, return_tensors='pt').to(model.device)
            input_size = inputs['input_ids'].shape[-1]
            outputs = model.generate(**inputs, do_sample=False, max_new_tokens=100)
            output_text = tokenizer.decode(outputs[0][input_size:], skip_special_tokens=True)
            print(output_text)
            if "support" in output_text.lower():
                pred_label = True
            elif "refute" in output_text.lower():
                pred_label = False
            else:
                pred_label = None
                
            if pred_label == label:
                correct += 1
            elif pred_label is not None:
                wrong += 1
        print(f"correct: {correct}, wrong: {wrong}, total: {total}, accuracy: {correct/total:.4f}")