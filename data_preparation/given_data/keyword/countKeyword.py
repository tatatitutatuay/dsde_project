import pandas as pd
from collections import Counter

# Load the CSV file
keywords_df = pd.read_csv('data_preparation\given_data\data\keywords.csv', on_bad_lines='skip')

# Flatten the DataFrame into a list of keywords, ignoring empty cells
keywords_list = keywords_df.values.ravel()  # Flatten the DataFrame into a 1D array
keywords_list = [kw for kw in keywords_list if kw]  # Remove empty values

# Count the frequency of each keyword
keyword_counts = Counter(keywords_list)

# Convert the Counter into a DataFrame for better readability
keyword_counts_df = pd.DataFrame(keyword_counts.items(), columns=['Keyword', 'Count'])

# Sort by count in descending order
keyword_counts_df = keyword_counts_df.sort_values(by='Count', ascending=False)

# Drop the row of Nan values
keyword_counts_df = keyword_counts_df.dropna()

# Save the results to a CSV
keyword_counts_df.to_csv('data_preparation\given_data\data\keyword_counts_2.csv', index=False)

# Display the result
print(keyword_counts_df)
