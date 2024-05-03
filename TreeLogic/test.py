import os
import re
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
folder = current_directory + '\\Decision Trees'
folder = folder.replace('\\', '/')
folder = './Decision Trees'

for file in os.listdir(folder):
        #check if file is a json file and call the function to add it to the warehouse
    if file.endswith('.json'):
        print(file)