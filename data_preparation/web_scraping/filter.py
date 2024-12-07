import pandas as pd

# Load the CSV file
df = pd.read_csv('data_preparation/web_scraping/merged_output.csv')

# Filter out rows with missing values
df = df.dropna()

# Filter out rows where the 'keyword' column contains empty or invalid lists
invalid_keywords = ['[]', '[""]', "['']"]
df = df[~df['keyword'].isin(invalid_keywords)]

# Drop the 'country' column
if 'country' in df.columns:
    df = df.drop(columns=['country'])

# Rename the 'keyword' column to 'keywords'
df = df.rename(columns={'keyword': 'keywords'})

# Save the filtered data to a new CSV file
df.to_csv('data_preparation/web_scraping/filtered_output.csv', index=False)
