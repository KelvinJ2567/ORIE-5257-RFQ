import pandas as pd
import hashlib
import json
from utils import hash_tuple


rfq_training_data = pd.read_excel('./data/rfq.xlsx', sheet_name='InSample')

next_mid_dict = {}

for index, row in rfq_training_data.iterrows():
    md5 = hashlib.md5()
    row['Quantity'] = int(row['Notional'] / 100)
    key = row[
        ['Bond', 'Side', 'Quantity', 'Counterparty', 'MidPrice', 'Competitors']
    ]
    key['Bond'] = key['Bond'].split(' ')[-1]  # Remove the 'US Treasury' prefix
    hash_value = hash_tuple(tuple(key.values))
    next_mid_dict[hash_value] = row['nextMidPrice']

# Save the next_mid_dict to a json file
with open('next_mid_dict.json', 'w') as f:
    json.dump(next_mid_dict, f)