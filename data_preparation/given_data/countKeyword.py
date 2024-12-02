import pandas as pd
import ast

df = pd.read_csv("data_preparation/given_data/data/data_noTHInAbstract.csv")

# Convert string representations of lists into actual lists
keywords = df['keywords'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x != '[]' else [])

# Flatten the list
flattened_keywords = [keyword for sublist in keywords for keyword in sublist]

# Count the occurrences of each keyword
keyword_counts = pd.Series(flattened_keywords).value_counts()

# Create an array with keyword and count pairs
keyword_array = [[keyword, count] for keyword, count in keyword_counts.items()]

# Export the array to a CSV file
output_file = "keyword_counts.csv"
keyword_df = pd.DataFrame(keyword_array, columns=['Keyword', 'Count'])
keyword_df.to_csv(output_file, index=False)

print(f"Data has been exported to {output_file}")
