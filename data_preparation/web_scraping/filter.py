import pandas as pd
import ast
import pycountry
import re

# Load the CSV file
df = pd.read_csv('data_preparation/web_scraping/scrape_output.csv')

# Filter out rows with missing values
df = df.dropna()

# Rename the 'keyword' column to 'keywords'
df = df.rename(columns={'keyword': 'keywords'})

# Function to clean the keywords
def clean_keywords(keywords):
    # Convert the string to list if it's not already a list
    if isinstance(keywords, str):
        keywords = ast.literal_eval(keywords)
    
    # Remove unwanted keywords
    unwanted_keywords = ['Index', 'Terms: ','Key words and phrases:', 'Key words and phrases: ', 'Key', 'Words', 'Words.: ']
    cleaned_keywords = [word for word in keywords if word not in unwanted_keywords]
    
    # Remove unwanted in part of a word 
    cleaned_keywords = [word.replace('Terms: \n', '').replace('/n', '').replace('Words.: \n', '').replace('Key words and phrases: ', '').replace(';', '').replace(' -', '') for word in cleaned_keywords]

    final_keywords = []
    for word in cleaned_keywords:
        # Split by commas and strip any extra whitespace
        if ',' in word:
            final_keywords.extend([w.strip() for w in word.split(',')])
        else:
            final_keywords.append(word)

    # Remove empty strings from the list
    final_keywords = [word for word in final_keywords if word != '']
    
    return final_keywords

# Function to clean and validate the country column using pycountry
def clean_country(country_list):
    if isinstance(country_list, str):
        country_list = ast.literal_eval(country_list)
    
    cleaned_country_list = []
    for country in country_list:
        cleaned_country = extract_valid_country(country)
        if cleaned_country:
            cleaned_country_list.append(cleaned_country)
    
    # Remove duplicates by converting to set and back to list
    cleaned_country_list = list(set(cleaned_country_list))
    
    return cleaned_country_list

def extract_valid_country(country_name):
    # Extract potential country names from strings like 'UniversityBostonUSA' or 'ResearchBeijingChina'
    country_name = re.sub(r'[^a-zA-Z ]', '', country_name)  # Remove non-alphabetic characters
    
    # Check all country names in pycountry
    for country in pycountry.countries:
        if country_name.endswith(country.name):
            return country.name  # Return only the valid country part
    
    return None  # If no valid country name is found, return None

invalid_keywords = ['[]', '[""]', "['']"]
df = df[~df['keywords'].isin(invalid_keywords)]
df['keywords'] = df['keywords'].apply(clean_keywords)
df['country'] = df['country'].apply(clean_country)

# Save the filtered data to a new CSV file
df.to_csv('data_preparation/web_scraping/filtered_output.csv', index=False)
