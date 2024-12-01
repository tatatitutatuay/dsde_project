import csv

def get_unique_keywords(csv_file_path):
    unique_keywords = set()
    
    with open(csv_file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader)
        keywords_index = header.index('keywords')
        
        for row in csvreader:
            keywords = row[keywords_index]
            keyword_list = keywords.strip('[]').split(',')
            for keyword in keyword_list:
                unique_keywords.add(keyword.strip().strip("'").strip('"'))
    
    return unique_keywords

if __name__ == "__main__":
    csv_file_path = 'dsde_project\data_preparation\given_data\more_filtered_scopus_data.csv'
    unique_keywords = get_unique_keywords(csv_file_path)
    with open('unique_keywords.txt', mode='w', encoding='utf-8') as file:
        for keyword in unique_keywords:
            file.write(f"{keyword}\n")
    print(unique_keywords)