import pandas as pd
import ast

df = pd.read_csv("filtered_scopus_data.csv")

# Convert string representations of lists into actual lists
keywords = df['keywords'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x != '[]' else [])

# Flatten the list
flattened_keywords = [keyword for sublist in keywords for keyword in sublist]

# Count the occurrences of each keyword
keyword_counts = pd.Series(flattened_keywords).value_counts()
print(keyword_counts.head(10))


# Count the number of unique keywords
unique_keywords = keyword_counts.index
#print(f"Number of unique keywords: {len(unique_keywords)}")


