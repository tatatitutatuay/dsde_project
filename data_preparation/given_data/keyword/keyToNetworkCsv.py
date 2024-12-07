import pandas as pd

# Step 1: Load the CSV files
keywords_df = pd.read_csv('data_preparation\given_data\keyword\keywords.csv', on_bad_lines='skip')
keyword_counts_df = pd.read_csv('data_preparation\given_data\keyword\keywords_count.csv')

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

# Filter out values less than 30 and the maximum value
filtered_output_df = output_df[(output_df['value'] >= 20) & (output_df['value'] != output_df['value'].max())]

# Step 4: Add connections
connections = []
for _, row in keywords_df.iterrows():
    row_keywords = [kw for kw in row if kw]  # Get all non-null keywords
    node_ids = [keyword_to_node_id[kw] for kw in row_keywords]  # Convert keywords to node ids
    
    for node_id in node_ids:

        if node_id not in filtered_output_df['node_id'].values:
            continue

        # Generate the connected node ids, excluding the current node_id
        connected_ids = sorted(set(nid for nid in node_ids if nid != node_id and nid in filtered_output_df['node_id'].values))  # No self-loops, sorted
        
        # If only one connected node, don't add a comma
        if len(connected_ids) == 1:
            connection = f"({connected_ids[0]})"
        else:
            connection = f"({', '.join(map(str, connected_ids))})"
        
        connections.append({"node_id": node_id, "connection": connection})  # Store connection as string

connections_df = pd.DataFrame(connections)
connections_df = connections_df.drop_duplicates(subset=["node_id", "connection"])

# Merge connections into the output DataFrame
output_df = filtered_output_df.merge(connections_df, on="node_id", how="left")

# Clean parentheses and convert to string
output_df['connection'] = output_df['connection'].astype(str).str.strip("()")  # Clean parentheses

# Group by node_id and aggregate connections
grouped_df = output_df.groupby(['node_id', 'keyword', 'value'], as_index=False).agg({
    'connection': lambda x: f"({', '.join(sorted(set(filter(None, ', '.join(x).split(', ')))))})"  # Clean and deduplicate connections
})

# Save the result
grouped_df.to_csv('data_preparation\\given_data\\keyword\\network.csv', index=False)
print("Successfully saved the consolidated output CSV file.")
