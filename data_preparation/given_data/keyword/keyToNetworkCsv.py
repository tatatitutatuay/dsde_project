import pandas as pd
from collections import defaultdict

# Step 1: Load the CSV files
keywords_df = pd.read_csv('data_preparation\given_data\keyword\keywords.csv', on_bad_lines='skip')
keyword_counts_df = pd.read_csv('data_preparation\given_data\keyword\keyword_counts.csv')

keywords_df = keywords_df.fillna("")

# Flatten keywords and assign node_id
unique_keywords = pd.unique(keywords_df.values.ravel('K'))
unique_keywords = [kw for kw in unique_keywords if kw]  # Remove empty strings
keyword_to_node_id = {kw: idx for idx, kw in enumerate(unique_keywords)}

# Create the node_id and keyword mapping
output_df = pd.DataFrame({
    "node_id": [keyword_to_node_id[kw] for kw in unique_keywords],
    "keyword": unique_keywords
})

# Map values from keyword_counts.csv
keyword_to_value = dict(zip(keyword_counts_df['Keyword'], keyword_counts_df['Count']))
output_df['value'] = output_df['keyword'].map(keyword_to_value).fillna(0)  # Fill missing values with 0

# Filter out values less than 20 and the maximum value
filtered_output_df = output_df[(output_df['value'] >= 20) & (output_df['value'] != output_df['value'].max())]

# Track connection frequencies
connection_frequencies = defaultdict(int)

# First pass: count how many times each pair appears together in rows
for _, row in keywords_df.iterrows():
    row_keywords = [kw for kw in row if kw]
    node_ids = [keyword_to_node_id[kw] for kw in row_keywords]
    node_ids = [nid for nid in node_ids if nid in filtered_output_df['node_id'].values]
    
    # Count pairs in this row
    for i in range(len(node_ids)):
        for j in range(i + 1, len(node_ids)):
            pair = tuple(sorted([node_ids[i], node_ids[j]]))
            connection_frequencies[pair] += 1

# Get pairs that appear at least twice
valid_pairs = {pair for pair, freq in connection_frequencies.items() if freq >= 2}

# Create connections dictionary for nodes with valid connections only
connections = {}
for node_id in filtered_output_df['node_id']:
    # Find all valid connections for this node
    connected_nodes = set()
    for pair in valid_pairs:
        if node_id in pair:
            other_node = pair[1] if pair[0] == node_id else pair[0]
            connected_nodes.add(other_node)
    
    # Only add nodes that have connections
    if connected_nodes:
        connections[node_id] = f"({', '.join(map(str, sorted(connected_nodes)))})"

# Create final DataFrame with only nodes that have connections
final_df = filtered_output_df[filtered_output_df['node_id'].isin(connections.keys())].copy()
final_df['connection'] = final_df['node_id'].map(connections)

# Save the result with proper quoting
final_df.to_csv('data_preparation\\given_data\\keyword\\network_test.csv', index=False, quoting=1)
print("Successfully saved the consolidated output CSV file.")