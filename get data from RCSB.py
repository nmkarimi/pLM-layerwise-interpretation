from rcsbsearchapi.search import SequenceQuery
import requests
import json
import time  
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import random

def result_SequenceQuery(sequence):
    results = SequenceQuery(sequence, 0.01, 0.7)
    return results

def fetch_sequence_details(polymer_entity_id):
    entry_id, entity_id = polymer_entity_id.split("_")
    url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{entry_id}/{entity_id}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        sequence = data.get('entity_poly', {}).get('pdbx_seq_one_letter_code_can', 'Sequence not found')
        chains = data.get('rcsb_polymer_entity_container_identifiers', {}).get('auth_asym_ids', [])
        # Select chain A or the first chain if A is not present

        if 'A' in chains:
            selected_chain = 'A'
        elif chains:
            selected_chain = chains[0]
        else:
            selected_chain = 'Chain not found'

        return sequence, selected_chain
    else:
        return f"Failed to fetch sequence for {polymer_entity_id}: {response.status_code}", 'Chain not found'


def process_element(element):
    try:
        results = result_SequenceQuery(element[2])
        result_items = []
        for polyid in results("polymer_entity"):
            sequence, chains = fetch_sequence_details(polyid)
            if len(sequence) < 1000:
              result_items.append([polyid.split('_')[0], chains, sequence])
        return result_items
    except Exception as e:
        print(f"Error processing element {element}: {e}")
        return []

def process_batch(batch):
    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_element = {executor.submit(process_element, element): element for element in batch}
        for future in as_completed(future_to_element):
            element = future_to_element[future]
            result_items = future.result()
            results.extend(result_items)
    return results


list_sequence_pdbID = []
list_sequence_pdbID_temporal = [
    ['1vr4', ['A', 'B', 'C', 'D', 'E'], "MIVTTTSGIQGKEIIEYIDIVNGEAIMGANIVRDLFASVRDVVGGRAGSYESKLKEARDIAMDEMKELAKQKGANAIVGVDVDYEVVRDGMLMVAVSGTAVRI"],
    ['1DG1', ['A','B'], "MSKEKFERTKPHVNVGTIGHVDHGKTTLTAAITTVLAKTYGGAARAFDQIDNAPEEKARGITINTSHVEYDTPTRHYAHVDCPGHADYVKNMITGAAQMDGAILVVAATDGPMPQTREHILLGRQVGVPYIIVFLNKCDMVDDEELLELVEMEVRELLSQYDFPGDDTPIVRGSALKALEGDAEWEAKILELAGFLDSYIPEPERAIDKPFLLPIEDVFSISGRGTVVTGRVERGIIKVGEEVEIVGIKETQKSTCTGVEMFRKLLDEGRAGENVGVLLRGIKREEIERGQVLAKPGTIKPHTKFESEVYILSKDEGGRHTPFFKGYRPQFYFRTTDVTGTIELPEGVEMVMPGDNIKMVVTLIHPIAMDDGLRFAIREGGRTVGAGVVAKVLS"],
    ['1D8T', ['A', 'B', 'C', 'D'], "SKEKFERTKPHVNVGTIGHVDHGKTTLTAAITTVLAKTYGGAARAFDQIDNAPEEKARGITINTSHVEYDTPTRHYAHVDCPGHADYVKNMITGAAQMDGAILVVAATDGPMPQTREHILLGRQVGVPYIIVFLNKCDMVDDEELLELVEMEVRELLSQYDFPGDDTPIVRGSALKALEGDAEWEAKILELAGFLDSYIPEPERAIDKPFLLPIEDVFSISGRGTVVTGRVERGIIKVGEEVEIVGIKETQKSTCTGVEMFRKLLDEGRAGENVGVLLRGIKREEIERGQVLAKPGTIKPHTKFESEVYILSKDEGGRHTPFFKGYRPQFYFRTTDVTGTIELPEGVEMVMPGDNIKMVVTLIHPIAMDDGLRFAIREGGRTVGAGVVAKVLG"],
    ['1TUI', ['A', 'B', 'C'], "AKGEFIRTKPHVNVGTIGHVDHGKTTLTAALTYVAAAENPNVEVKDYGDIDKAPEERARGITINTAHVEYETAKRHYSHVDCPGHADYIKNMITGAAQMDGAILVVSAADGPMPQTREHILLARQVGVPYIVVFMNKVDMVDDPELLDLVEMEVRDLLNQYEFPGDEVPVIRGSALLALEEMHKNPKTKRGENEWVDKIWELLDAIDEYIPTPVRDVDKPFLMPVEDVFTITGRGTVATGRIERGKVKVGDEVEIVGLAPETRKTVVTGVEMHRKTLQEGIAGDNVGLLLRGVSREEVERGQVLAKPGSITPHTKFEASVYILKKEEGGRHTGFFTGYRPQFYFRTTDVTGVVRLPQGVEMVMPGDNVTFTVELIKPVALEEGLRFAIREGGRTVGAGVVTKILE"],
    ['1EFT', ['A'], "AKGEFIRTKPHVNVGTIGHVDHGKTTLTAALTFVTAAENPNVEVKDYGDIDKAPEERARGITINTAHVEYETAKRHYSHVDCPGHADYIKNMITGAAQMDGAILVVSAADGPMPQTREHILLARQVGVPYIVVFMNKVDMVDDPELLDLVEMEVRDLLNQYEFPGDEVPVIRGSALLALEEMHKNPKTKRGENEWVDKIWELLDAIDEYIPTPVRDVDKPFLMPVEDVFTITGRGTVATGRIERGKVKVGDEVEIVGLAPETRKTVVTGVEMHRKTLQEGIAGDNVGLLLRGVSREEVERGQVLAKPGSITPHTKFEASVYVLKKEEGGRHTGFFSGYRPQFYFRTTDVTGVVRLPQGVEMVMPGDNVTFTVELIKPVALEEGLRFAIREGGRTVGAGVVTKILE"]
]

while list_sequence_pdbID_temporal and len(list_sequence_pdbID) <= 1000:
    # Process batch of elements from list_sequence_pdbID_temporal
    batch = list_sequence_pdbID_temporal[:10]  # Batch size of 10
    result_items = process_batch(batch)
    list_sequence_pdbID_temporal = list_sequence_pdbID_temporal[10:]  # Remove processed elements

    # Add new results to list_sequence_pdbID
    list_sequence_pdbID.extend(result_items)

    # Debugging output
    print(f"Number of items in list_sequence_pdbID: {len(list_sequence_pdbID)}")

    # Optional: sleep to avoid overloading API
    time.sleep(15)


db = list_sequence_pdbID

data = []
for i in range(len(db)):
    data.append({'entry_id': db[i][0], 'asym_id': db[i][1]})


pairs = [(data[i], data[j]) for i in range(len(data)) for j in range(i+1, len(data))]


random.shuffle(pairs)

import requests
import json
import time
import pandas as pd

BATCH_SIZE = 1  

def send_request(entry_ids, asym_ids):

    structures = [
        {
            "entry_id": entry_id,
            "selection": {
                "asym_id": asym_id
            }
        }
        for entry_id, asym_id in zip(entry_ids, asym_ids)
    ]

    query = {
        "context": {
            "mode": "pairwise",
            "method": {
                "name": "smith-waterman-3d",
                "parameters": {
                    "gap_opening_penalty": 8,
                    "gap_extension_penalty": 1
                }
            },
            "structures": structures
        }
    }

    data = {"query": json.dumps(query)}
    try:
        response = requests.post("https://alignment.rcsb.org/api/v1/structures/submit", data=data)
        response.raise_for_status()
        return response.text.strip()
    except requests.RequestException:
        return None


def check_job_status(job_ticket):
    url = f"https://alignment.rcsb.org/api/v1/structures/results?uuid={job_ticket}"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        status = data.get('info', {}).get('status', 'RUNNING')
        return status, data
    except requests.RequestException:
        return 'ERROR', None


def func_asli(entry_id1, asym_id1, entry_id2, asym_id2):
    entry_ids = [entry_id1, entry_id2]
    asym_ids = [asym_id1, asym_id2]

    job_ticket = send_request(entry_ids, asym_ids)

    if not job_ticket:
        return None  

    status = 'RUNNING'
    while status == 'RUNNING':
        status, data = check_job_status(job_ticket)
        if status == 'RUNNING':
            time.sleep(5)  
        elif status == 'COMPLETE':
            return extract_info(data)
        elif status == 'ERROR':
            return {"status": "ERROR", "data": data}


def extract_info(data):
    results = data.get('results', [])
    if not results:
        return None

    result = results[0]
    sequence_alignments = result.get('sequence_alignment', [])
    sequences = [alignment['sequence'] for alignment in sequence_alignments]

    sequence_summary_scores = result.get('summary', {}).get('scores', [])
    sequence_summary_scores = {score['type']: score['value'] for score in sequence_summary_scores}

    if sequence_summary_scores.get('sequence-similarity') > 0.70 and sequence_summary_scores.get('RMSD') > 3 and sequences[0] != sequences[1] and len(sequences[0]) < 1000 and len(sequences[1]) < 1000:
      return [sequences[0], sequences[1], sequence_summary_scores.get('sequence-similarity'),
            sequence_summary_scores.get('TM-score'), sequence_summary_scores.get('sequence-identity'),
            sequence_summary_scores.get('RMSD'), sequence_summary_scores.get('sequence-similarity')]
    else:
      return []

unique_pairs = set()

def is_unique_pair(row):
    sorted_pair = tuple(sorted([row['entry_id_0'], row['entry_id_1']]))
    if sorted_pair in unique_pairs:
        return False
    unique_pairs.add(sorted_pair)
    return True


def process_sequentially(pairs):
    results = []
    index = 0
    for pair in pairs:
        index = index + 1
        if index % 50 == 0:
          df_results = pd.DataFrame(results, columns=['entry_id_0', 'asym_id_0', 'entry_id_1', 'asym_id_1', "sequence1", "sequence2", "similarity_score", "TM_score", "sequence_identity", "RMSD", "sequence_similarity"])
          with pd.ExcelWriter('/home/rezaeian/Documents/mkarimi/Smith-waterman/result_get_data_RCSB.xlsx') as writer:
            df_results.to_excel(writer)
            
        result1 = [pair[0]['entry_id'].upper(), pair[0]['asym_id'].upper(), pair[1]['entry_id'].upper(), pair[1]['asym_id'].upper()]
        result = func_asli(pair[0]['entry_id'].upper(), pair[0]['asym_id'].upper(), pair[1]['entry_id'].upper(), pair[1]['asym_id'].upper())

        if result and len(result) == 7:
          for res in result:
             result1.append(res)
          result = result1
          if result:
            results.append(result)
            print(len(results))
            if len(results) == 2000:
              return results


start_time = time.time()
results = process_sequentially(pairs) 
end_time = time.time()


df_results = pd.DataFrame(results, columns=['entry_id_0', 'asym_id_0', 'entry_id_1', 'asym_id_1', "sequence1", "sequence2", "similarity_score", "TM_score", "sequence_identity", "RMSD", "sequence_similarity"])
with pd.ExcelWriter('/home/rezaeian/Documents/mkarimi/Smith-waterman/result_get_data_RCSB.xlsx') as writer:
      df_results.to_excel(writer)

execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")
