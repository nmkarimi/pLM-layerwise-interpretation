
import pandas as pd
import json
import csv
from transformers import T5Tokenizer, T5EncoderModel,T5Config
import torch
import re

config = T5Config.from_pretrained("Rostlab/prot_t5_xl_uniref50")
config.output_hidden_states = True

#tokenizer = T5Tokenizer.from_pretrained("Rostlabprot_t5_xl_uniref50",use_fast=False,do_lower_case=False)
#model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50", config=config)
#model = model.eval().to("cuda" if torch.cuda.is_available() else "cpu")


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(device)
def t5_embedding(sequence1,sequence2, layer_num=24):
  #seq = [sequence1, sequence2]
  if device == torch.device("cpu"):
    model.to(torch.float32)
  sequence_examples = [sequence1, sequence2]
  len_seq=[len(sequence_examples[0]),len(sequence_examples[1])]

  sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]

  ids = tokenizer(sequence_examples, add_special_tokens=True, padding="longest")

  input_ids = torch.tensor(ids['input_ids']).to(device)
  attention_mask = torch.tensor(ids['attention_mask']).to(device)

  with torch.no_grad():
    embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

  layer_hidden_states = embedding_repr.hidden_states[layer_num]
  emb_0 = layer_hidden_states[0,:len_seq[0]].tolist()
  emb_1 = layer_hidden_states[1,:len_seq[1]].tolist()
  embedding_list=[]
  embedding_list.append(emb_0)
  embedding_list.append(emb_1)

  return embedding_list

df_results_smith_waterman_3 = pd.read_excel('/home/mehdi/Documents/revise_hypo_2/filter_get_data_RCSB_with_TM.xlsx')
df_results_smith_waterman_3['embedding_0'] = None
df_results_smith_waterman_3['embedding_1'] = None


for kk in range(int(1133/50)+1):
  print(kk)
  print('---------------------------------')
  output_file = "/home/mehdi/Documents/revise_hypo_2/t5/24/processed_embedding_T5_"+str(kk)+".csv"


  try:
    with open(output_file, 'x', newline='') as f:
        writer = csv.writer(f)
  
        writer.writerow([
            'entry_id_0', 'asym_id_0', 'entry_id_1', 'asym_id_1',
            "sequence1", "sequence2", "similarity_score", "TM_score",
            "sequence_identity", "RMSD", "sequence_similarity",
            "embedding_0", "embedding_1"
        ])
  except FileExistsError:
    pass

  with open(output_file, 'a', newline='') as f:
    writer = csv.writer(f)

    for index, row in df_results_smith_waterman_3.iterrows():
      if index>=kk*50:
        #print(index)

        out = t5_embedding(row['sequence1'], row['sequence2'])

        embed1_json = json.dumps(out[0])
        embed2_json = json.dumps(out[1])

        writer.writerow([
            row['entry_id_0'], row['asym_id_0'], row['entry_id_1'], row['asym_id_1'],
            row["sequence1"], row["sequence2"], row["similarity_score"],
            row["TM_score"], row["sequence_identity"], row["RMSD"],
            row["sequence_similarity"], embed1_json, embed2_json
        ])
        if (index+1)%50==0:
          break


