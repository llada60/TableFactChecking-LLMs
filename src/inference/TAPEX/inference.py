from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch
import json
import argparse
import time

tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/tapex-large-finetuned-tabfact"
)
model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/tapex-large-finetuned-tabfact"
)
model.eval()

def table_str_to_df(table_str):
    rows = [r.split("#") for r in table_str.strip().split("\n")]
    header, data = rows[0], rows[1:]
    return pd.DataFrame(data, columns=header)

def tapex_predict(table_str, statement):
    table_df = table_str_to_df(table_str)

    encoding = tokenizer(
        table=table_df,
        query=statement,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**encoding)

    pred_idx = outputs.logits.argmax(dim=-1).item()
    pred_label = model.config.id2label[pred_idx]

    return 1 if pred_label == "Entailed" else 0

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, required=True, help="Path to the JSON dataset file.")
parser.add_argument("--max_runs", type=int, default=-1, help="Maximum number of runs to execute.")

args = parser.parse_args()
dataset_path = args.dataset_path
max_runs = args.max_runs

dataset = json.load(open(dataset_path))

correct = 0
total = 0

n=0

start_time = time.time()

for ex in dataset:
    if n==max_runs:
        break
    n+=1
    pred = tapex_predict(ex["table"], ex["statement"])
    gold = ex["label"]

    print(f"Statement: {ex['statement']}")
    print(f"Predicted: {pred}, Gold: {gold}\n")

    correct += (pred == gold)
    total += 1
end_time = time.time()
print(f"TAPEX accuracy: {correct/total * 100:.2f}% ({correct}/{total})")
print(f"Total inference time: {end_time - start_time:.2f} seconds")