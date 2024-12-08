import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType
import os
import shutil

# Initialize Spark session
spark = SparkSession.builder.appName("Combine CSVs").getOrCreate()

# combine two csv files
df1 = pd.read_csv('data_preparation/web_scraping/filtered_output.csv')
df2 = pd.read_csv('data_preparation/given_data/data/data_noTHInAbstract.csv')

df1 = df1.drop(columns=['country'])

combined_df = pd.concat([df1, df2], ignore_index=True)

schema = StructType([
    StructField("abstract", StringType(), True),
    StructField("keywords", StringType(), True),
])

df = spark.createDataFrame(combined_df, schema)

# Save the merged data to a new CSV file
output_dir = 'data_preparation/temp'
single_file = 'data_preparation/trainData.csv'

df.coalesce(1).write.csv(output_dir, header=True, mode='overwrite')

for file_name in os.listdir(output_dir):
    if file_name.endswith(".csv"):
        shutil.move(os.path.join(output_dir, file_name), single_file)
shutil.rmtree(output_dir)
