import pandas as pd
import ast

# Read the data
df = pd.read_csv("data_preparation\given_data\data\more_filtered_scopus_data.csv")

# Convert string representations of lists into actual lists
keywords = df['keywords'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x != '[]' else [])

# Flatten the list to find the most frequent keyword
flattened_keywords = [keyword for sublist in keywords for keyword in sublist]

# Count the occurrences of each keyword
keyword_counts = pd.Series(flattened_keywords).value_counts()

# Get the most frequent keyword (number 1 keyword)
most_frequent_keyword = keyword_counts.index[0]

# Remove the most frequent keyword from each list in the 'keywords' column when the keyword is not in the abstract
df['keywords'] = df.apply(
    lambda row: [keyword for keyword in ast.literal_eval(row['keywords']) if keyword != most_frequent_keyword] 
    if isinstance(row['keywords'], str) and row['keywords'] != '[]' and most_frequent_keyword not in row['abstract'] else ast.literal_eval(row['keywords']),
    axis=1
)
# Save the modified DataFrame back to a CSV file (optional)
df.to_csv("modified_data.csv", index=False)

# Optionally print the modified DataFrame to verify
print(df.head())
