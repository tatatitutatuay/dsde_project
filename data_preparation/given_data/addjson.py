import os

directory_path = './Project/2023'  # Set your directory path here

for filename in os.listdir(directory_path):
    file_path = os.path.join(directory_path, filename)
    if os.path.isfile(file_path):
        new_file_path = os.path.join(directory_path, os.path.splitext(filename)[0] + '.json')
        os.rename(file_path, new_file_path)
        print(f'Renamed {filename} to {os.path.splitext(filename)[0]}.json')
