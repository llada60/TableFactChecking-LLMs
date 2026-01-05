import argparse


from tensorflow.python.client import device_lib
from tqdm import tqdm


# In[3]:


import tensorflow.compat.v1 as tf
import os 
import shutil
import csv
import pandas as pd
import IPython

tf.get_logger().setLevel('ERROR')


# In[4]:


from tapas.utils import tf_example_utils
from tapas.protos import interaction_pb2
from tapas.utils import number_annotation_utils
import math
import json

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='./data/test_examples_with_csv_paraphrased.json', help='Path to the input JSON data file')
parser.add_argument('--output_path', type=str, default='./logs/tapas', help='Path to save the predictions')
parser.add_argument('--model_path', type=str, default='./checkpoints/tapas/tapas_tabfact_inter_masklm_large_reset', help='Name of the pre-trained model to use')
args = parser.parse_args()


max_seq_length = 512
vocab_file = os.path.join(args.model_path, 'vocab.txt')
config = tf_example_utils.ClassifierConversionConfig(
    vocab_file=vocab_file,
    max_seq_length=max_seq_length,
    max_column_id=max_seq_length,
    max_row_id=max_seq_length,
    strip_column_names=False,
    add_aggregation_candidates=False,
)
converter = tf_example_utils.ToClassifierTensorflowExample(config)

def convert_interactions_to_examples(tables_and_queries):
  """Calls Tapas converter to convert interaction to example."""
  for idx, (table, queries) in enumerate(tables_and_queries):
    interaction = interaction_pb2.Interaction()
    for position, query in enumerate(queries):
      question = interaction.questions.add()
      question.original_text = query
      question.id = f"{idx}-0_{position}"
    for header in table[0]:
      interaction.table.columns.add().text = header
    for line in table[1:]:
      row = interaction.table.rows.add()
      for cell in line:
        row.cells.add().text = cell
    number_annotation_utils.add_numeric_values(interaction)
    for i in range(len(interaction.questions)):
      try:
        yield converter.convert(interaction, i)
      except ValueError as e:
        print(f"Can't convert interaction: {interaction.id} error: {e}")
        
def write_tf_example(filename, examples):
  with tf.io.TFRecordWriter(filename) as writer:
    for example in examples:
      writer.write(example.SerializeToString())

def predict(table_data, queries, args, verbose=False):
  table = [list(map(lambda s: s.strip(), row.split("#"))) 
         for row in table_data.strip().split("\n") if row.strip()]
  
  examples = convert_interactions_to_examples([(table, queries)])
  write_tf_example(os.path.join(args.output_path, "tabfact/tf_examples/test.tfrecord"), examples)
  write_tf_example(os.path.join(args.output_path, "tabfact/tf_examples/dev.tfrecord"), [])
  
  import subprocess

  cmd = [
      "python", "-m", "tapas.run_task_main",
      '--task=TABFACT',
      f'--output_dir={args.output_path}',
      '--noloop_predict',
      f'--test_batch_size={len(queries)}',
      '--tapas_verbosity=ERROR',
      '--compression_type=',
      '--reset_position_index_per_cell',
      f'--init_checkpoint={args.model_path}/model.ckpt',
      f'--bert_config_file={args.model_path}/bert_config.json',
      '--mode=predict',
  ]

  with open("error", "w") as err:
      subprocess.run(cmd, stderr=err, check=True)

  results_path = os.path.join(args.output_path, 'tabfact/model/test.tsv')
  all_results = []
  if verbose:
      df = pd.DataFrame(table[1:], columns=table[0])
      display(IPython.display.HTML(df.to_html(index=False)))
  
  with open(results_path) as csvfile:
      reader = csv.DictReader(csvfile, delimiter='\t')
      for row in reader:
          supported = int(row["pred_cls"])
          all_results.append(supported)
          if verbose:
              position = int(row['position'])
              if supported:
                  print("> SUPPORTS:", queries[position])
              else:
                  print("> REFUTES:", queries[position])
  return all_results


# # Predict




dataset_original = json.load(open(args.data_path, 'r'))
args.output_path = os.path.join(args.output_path, os.path.basename(args.data_path).replace('.json', ''))
os.makedirs(args.output_path, exist_ok=True)
os.makedirs(os.path.join(args.output_path, 'tabfact'), exist_ok=True)
print(args.output_path)
os.makedirs(os.path.join(args.output_path, 'tabfact/tf_examples'), exist_ok=True)
os.makedirs(os.path.join(args.output_path, 'tabfact/model'), exist_ok=True)
with open(os.path.join(args.output_path, 'tabfact/model/checkpoint'), 'w') as f:
  f.write('model_checkpoint_path: "model.ckpt-0"')
for suffix in ['.data-00000-of-00001', '.index', '.meta']:
  # f'../../../checkpoints/tapas/tapas_tabfact_inter_masklm_large_reset/model.ckpt{suffix}', f'results/tabfact/model/model.ckpt-0{suffix}'
  shutil.copyfile(os.path.join(args.model_path, f'model.ckpt{suffix}'), 
                  os.path.join(args.output_path, f'tabfact/model/model.ckpt-0{suffix}'))

all_preds_original = []
all_labels_original = []

pbar = tqdm(dataset_original.items(), total=len(dataset_original))
for key, value in pbar:
    questions, labels, entity, table_csv = value
    
    # prediction
    preds = predict(table_csv, questions, args, verbose=False)

    all_preds_original.extend(preds)
    all_labels_original.extend(labels)


correct = sum([p==l for p,l in zip(all_preds_original, all_labels_original)])
total = len(all_labels_original)
accuracy = correct / total

print(f"Tapas accuracy for original dataset: {accuracy*100:.2f}% ({correct}/{total})")

