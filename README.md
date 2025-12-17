# TableFactChecking-LLMs

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
