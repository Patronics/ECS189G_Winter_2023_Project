# -- compatibility layer --
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

sys.path.insert(0, ROOT_DIR)
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# -----------------------

import torch
from codes.stage_5_code.Dataset_Loader_Node_Classification import Dataset_Loader
DATASET_NAME = 'cora'
dataset_loader = Dataset_Loader(DATASET_NAME, '')
dataset_loader.dataset_name = DATASET_NAME
dataset_loader.dataset_source_file_name = DATASET_NAME #not used
dataset_loader.dataset_source_folder_path = f'../../data/stage_5_data/{DATASET_NAME}'
dataset = dataset_loader.load()
print()
