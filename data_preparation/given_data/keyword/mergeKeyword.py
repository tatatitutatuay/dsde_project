import pandas as pd

# Read the CSV files
df1 = pd.read_csv('data_preparation/given_data/data/data_noTHInAbstract.csv')
df2 = pd.read_csv('data_preparation/web_scraping/filtered_output.csv')

df1 = df1['keywords']
df2 = df2['keywords']

# Clean the keywords by removing unwanted characters
def clean_keyword(entry):
    entry = entry.replace('[', '').replace(']', '').replace("'", "").replace(' —', '').replace(':', '')
    
    # remove case sensitivity
    entry = entry.lower()
    
    # Strip leading/trailing spaces
    return entry.strip()

# Split each string by comma 
def split_keywords(entry):
    parts = [clean_keyword(part) for part in entry.split(',')]
    
    if len(parts) == 1:
        parts.append('')
    
    return parts

df1 = df1.apply(split_keywords)
df2 = df2.apply(split_keywords)

# Combine the split data from both DataFrames
df = pd.concat([df1, df2], ignore_index=True)

# Convert the combined data to a DataFrame
df = pd.DataFrame(df.tolist())

# Save the merged and split data to a new CSV file
df.to_csv('data_preparation/given_data/keyword/keywords.csv', index=False)
