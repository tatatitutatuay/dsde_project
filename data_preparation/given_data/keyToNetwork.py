import pandas as pd
import json

# Load the CSV file
csv_file = "data_preparation/given_data/data/keywords.csv"  # Replace with your CSV file path
df = pd.read_csv(csv_file, header=None, on_bad_lines='skip')  # Read without headers

# Lists to store all nodes and edges
all_nodes = []
all_edges = []

# Loop through the first 500 rows (from row 1 onwards)
for row_index in range(1, 21):
    # Get the keywords for the current row
    keywords = df.iloc[row_index].dropna().tolist()  # Drop NaN values, convert to list

    # Create nodes
    row_nodes = [{"id": len(all_nodes) + idx + 1, "label": keyword} for idx, keyword in enumerate(keywords)]
    all_nodes.extend(row_nodes)  # Append nodes of this row to all_nodes

    # Create edges (all-to-all connections within this row)
    row_edges = []
    for i in range(len(row_nodes)):
        for j in range(i + 1, len(row_nodes)):
            row_edges.append({"from": row_nodes[i]["id"], "to": row_nodes[j]["id"]})
    all_edges.extend(row_edges)  # Append edges of this row to all_edges

# Save the combined nodes and edges to a JSON file
output_file = "data_preparation/given_data/data/network_data_combined.json"
with open(output_file, "w") as f:
    json.dump({"nodes": all_nodes, "edges": all_edges}, f, indent=4)

print(f"Network data for 500 rows combined and saved to {output_file}")

