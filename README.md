# TableFactChecking-LLMs

## Running

### LLM
#### Serial_prompt

``` shell
python src/baselines/LLM/llm_json.py --data_path "./data/flattened_test_examples_with_csv_paraphrased.json" --model_name "Qwen/Qwen2.5-7B-Instruct-1M" --max_new_tokens 10 --temperature 0.0
```

#### Parallel_prompt

``` shell
python src/baselines/LLM/llm_parallel.py --data_path "./data/prompteDataset/test_examples_with_csv_direct_prompt.json" --model_name "Qwen/Qwen2.5-7B-Instruct-1M" --max_new_tokens 10 --temperature 0.0
```

#### Chat_mode

``` shell
python src/baselines/LLM/llm_json_chat.py --data_path "./data/flattened_test_examples_with_csv_paraphrased.json" --model_name "Qwen/Qwen2.5-7B-Instruct-1M" --max_new_tokens 10 --temperature 0.0
```

## Dataset
- The **original TabFact** is in `./data/test_examples_with_csv.json`
- The **paraphrased TabFact** is in `./data/test_examples_with_csv_paraphrased.json`
- The **adversarial TabFact** is in `./data/test_examples_with_csv_adversarial.json`
  
| Dataset | #Statemetents | #Tables | #True Statements |
|---------|---------------|---------|-----------------|
| Original TabFact | 12779 | 1695 | 6425 |
| Paraphrased TabFact | 12779 | 1695 | 6425 |
| Adversarial TabFact | 10928 | 1366 | 5462 |

The data format is as follows:
```json
{
  "csv_name.csv": [
    statements,
    labels,
    table_title,
    table
  ]
}
```
Field descriptions:
- `csv_name.csv`(string)
- `statements`(List[string]): statements to be verified against the table
- `labels`(List[int]): labels corresponding to each statement, either 0 (False) or 1 (True)
- `table_title`(string): title of the table
- `table`(string): CSV formatted table content

## Baselines

### LLM-based Methods

#### Trying Different Prompting Methods
1. **Direct Prompting**: Use direct prompts to query LLMs for fact-checking.
2. **Chat mode**: Use chat-based prompts to interact with LLMs for fact-checking.

#### Evaluating Different LLMs

Direct Prompting evaluated on the following LLMs:

1. `Qwen/Qwen2.5-7B-Instruct-1M`

2. `togethercomputer/Llama-2-7B-32K-Instruct`

3. `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct`

Chat mode evaluated on the following LLMs:

1. `Qwen/Qwen2.5-7B-Instruct-1M`

2. `meta-llama/Llama-2-7b-chat-hf`

3. `deepseek-ai/DeepSeek-V2-Lite-Chat`

## Data Preprocessing

### 1. Generate paraphrasing and adversarial statements

- **Paraphrasing**: Use LLM to generate paraphrased statements based on the original statements(same meaning but different wording). Here we challenged the model's robustness to different phrasings of the same statement.

- **Adversarial**: generate adversarial statements which have the same wording of statement A and B. The new statements are in the format of "A AND B" or "A OR B". Here we challenged the model's ability to reason consider the logical relationships between statements.

Generate paraphrased statements with llm:
``` shell
python src/data_generation/paraphrasing_statements_generation.py --data_path "./data/test_examples_with_csv.json" --model_name "Qwen/Qwen2.5-7B-Instruct-1M" --max_new_tokens 512 --temperature 0.7 --batch_size 4
```

Generate adversarial statements with llm:
``` shell
python src/data_generation/adversarial_statements_generation.py --data_path "./data/test_examples_with_csv.json" --model_name "Qwen/Qwen2.5-7B-Instruct-1M" --max_new_tokens 512 --temperature 0.7 --batch_size 4
```

Generate adversarial statements manually with "AND" and "OR":
``` shell
python src/data_generation/adversarial_statements_generation_manual.py --data_path "./data/test_examples_with_csv.json" 
```




### 2. Flatten the data structure and convert into Dataloader format

``` shell
python src/data_preprocessing/json_transfer.py --data_path "./data/test_examples_with_csv.json" --prompt_type "direct_prompt"
```

This will generate two files:
- `test_examples_with_csv_flattened.json`(List[Dict]): flattened data structure with keys: 'id', 'statement', 'label', 'table_title', 'table'
- `test_examples_with_csv_direct_prompt.json` (List[Dict]): prompted data for LLM input with keys: 'id', 'prompt', 'label'
