import json
import pandas as pd
import pickle
import os


with open('config.json','r') as file:
    fonfig_json = json.load(file)
    
local_data_dir = config_json['LOCAL_DATA_DIR']

#Өгөгдөл хадгалах folder нээх
if not os.path.exists(local_data_dir):
    os.mkdir(local_data_dir)
else:
    pass