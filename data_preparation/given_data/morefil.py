import pandas as pd
import json
import glob

# Path to your JSON files
json_files = glob.glob("./Project/2018/*.json")

# Add JSON files from other years
json_files.extend(glob.glob("./Project/2019/*.json"))
json_files.extend(glob.glob("./Project/2020/*.json"))
json_files.extend(glob.glob("./Project/2021/*.json"))
json_files.extend(glob.glob("./Project/2022/*.json"))
json_files.extend(glob.glob("./Project/2023/*.json"))


all_data = []

for file in json_files:
    with open(file, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            
            
            # Check if the required fields are present in the JSON structure
            coredata = data.get("abstracts-retrieval-response", {}).get("coredata", {})
            bibrecord = data.get("abstracts-retrieval-response", {}).get("item", {}).get("bibrecord", {})
            head = bibrecord.get("head", {})
            abstracts_retrieval_response = data.get("abstracts-retrieval-response", {})

            # Check if keywords are a list
            keywords = head.get("citation-info", {}).get("author-keywords", {}).get("author-keyword", [])
            if not isinstance(keywords, list):
                # drop if keywords are null
                if keywords is None:
                    continue
                
                # if key is only one, convert to list
                if isinstance(keywords, str):
                    keywords = [keywords]
                    
                print(f"File {file} contains a single keyword and was converted to a list.")
            

            if coredata and bibrecord and head:
                # Extract features
                features = {
                    "abstract": head.get("abstracts", None),
                    "keywords": [kw["$"] for kw in keywords if isinstance(kw, dict)],
                }
                
                # Append features if they meet the structure
                all_data.append(features)
            else:
                print(f"File {file} does not contain the required structure and was skipped.")
        
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Error processing file {file}: {e}")

# Convert to DataFrame
df = pd.DataFrame(all_data)

df = df.dropna(subset=['abstract', 'keywords'])

# drop if keyword is []
df = df[df['keywords'].map(lambda d: len(d)) > 0]

# Save as CSV
df.to_csv("more_filtered_scopus_data.csv", index=False)
