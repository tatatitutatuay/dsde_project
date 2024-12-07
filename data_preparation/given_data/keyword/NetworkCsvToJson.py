import pandas as pd
import ast
import json

def normalize(value, old_min, old_max, new_min, new_max):
    return ((value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min

# Load the network.csv
network_df = pd.read_csv('data_preparation\\given_data\\keyword\\network.csv')

# Prepare a list of nodes
nodes = []
for _, row in network_df.iterrows():
    node_id = row['node_id']
    keyword = row['keyword']
    value = row['value']

    # Normalize the value to a range of 10 to 100
    value = normalize(value, 20, network_df['value'].max(), 12, 60)
    
    # Add the node to the nodes list
    nodes.append({
        'id': node_id,
        'label': keyword,
        'size': int(value),
        'font' : { 'size': int(value) }
    })

# Prepare a list of edges
edges = []
for _, row in network_df.iterrows():
    node_id = row['node_id']
    connection_str = row['connection']
    
    # If the connection is not empty, evaluate it
    if pd.notna(connection_str) and connection_str != "":
        connections = ast.literal_eval(connection_str)  # Convert string of node_ids to a list
        
        # Ensure connections is a list or tuple, and iterate over them
        if isinstance(connections, (list, tuple)):
            for connected_id in connections:
                if connected_id > node_id:  # Avoid duplicate edges
                    edges.append({
                        'from': node_id,
                        'to': connected_id
                    })

# Prepare the final data in a JSON format
network_data = {
    'nodes': nodes,
    'edges': edges
}

# Save the data to a JSON file
with open('data_preparation\\given_data\\keyword\\network_data.json', 'w') as json_file:
    json.dump(network_data, json_file, indent=4)

print("Successfully saved the network data in JSON format.")
