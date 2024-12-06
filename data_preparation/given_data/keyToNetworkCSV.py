import pandas as pd

# Step 1: Load the CSV files
keywords_df = pd.read_csv('data_preparation\given_data\data\keywords.csv', on_bad_lines='skip')
keyword_counts_df = pd.read_csv('data_preparation\given_data\data\keyword_counts_2.csv')

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


# Step 4: Add connections
connections = []
for _, row in keywords_df.iterrows():
    row_keywords = [kw for kw in row if kw]
    node_ids = [keyword_to_node_id[kw] for kw in row_keywords]
    # print(row_keywords, node_ids)
    for node_id in node_ids:
        connected_ids = tuple(sorted(set(nid for nid in node_ids if nid != node_id)))  # Ensure tuple, sorted, and no self-loops
        connections.append({"node_id": node_id, "connection": connected_ids})  # Store as tuple for hashability

connections_df = pd.DataFrame(connections)
connections_df = connections_df.drop_duplicates(subset=["node_id", "connection"])

# Merge connections into the output DataFrame
output_df = output_df.merge(connections_df, on="node_id", how="left")

# Clean the connection column and ensure all values are strings
output_df['connection'] = output_df['connection'].astype(str).str.strip("()")  # Clean parentheses and convert to string

# Group by node_id and aggregate connections
grouped_df = output_df.groupby(['node_id', 'keyword', 'value'], as_index=False).agg({
    'connection': lambda x: f"({', '.join(sorted(set(', '.join(x).split(', '))))})"  # Combine and deduplicate connections
})

# Save the result
grouped_df.to_csv('data_preparation\given_data\data\\network.csv', index=False)
print("Successfully saved the consolidated output CSV file.")
