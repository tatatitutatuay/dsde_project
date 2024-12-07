import pandas as pd

# combine two csv files
df1 = pd.read_csv('data_preparation/web_scraping/filtered_output.csv')
df2 = pd.read_csv('data_preparation\given_data\data\data_noTHInAbstract.csv')

df = pd.concat([df1, df2], ignore_index=True)

# Save the merged data to a new CSV file
df.to_csv('data_preparation/trainData.csv', index=False)
