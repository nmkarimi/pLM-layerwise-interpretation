import pandas as pd
from openpyxl import Workbook, load_workbook
import json
import torch
import esm

# Set to track observed pairs regardless of their order
unique_pairs = set()

def is_unique_pair(row):
    """
    Check if a protein pair is unique by sorting IDs alphabetically.
    This effectively eliminates symmetric duplicates like (A, B) and (B, A).
    """
    sorted_pair = tuple(sorted([row['entry_id_0'], row['entry_id_1']]))
    if sorted_pair in unique_pairs:
        return False
    unique_pairs.add(sorted_pair)
    return True

# Load RCSB query results
sequences_df = pd.read_excel('/home/rezaeian/Documents/mkarimi/Smith-waterman/result_get_data_RCSB.xlsx')

# Filter the dataframe to retain only unique protein pairs
filtered_df = sequences_df[sequences_df.apply(is_unique_pair, axis=1)]

# Output the total count of retained unique pairs
print(len(filtered_df))

# Save the filtered dataset to an Excel file
with pd.ExcelWriter('/home/rezaeian/Documents/mkarimi/Smith-waterman/filter_get_data_RCSB.xlsx') as writer:
    filtered_df.to_excel(writer)
