# load_data.py
import pandas as pd

# Load the MedQuAD dataset
data = pd.read_json('medquads.json')

# Display the first few rows of the dataset
print(data.head())