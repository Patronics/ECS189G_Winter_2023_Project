# -- compatibility layer --
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

sys.path.insert(0, ROOT_DIR)
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# -----------------------

# -- configuration variables -- #
useGPU = True
DATASET_NAME = 'ORL'


if DATASET_NAME == 'ORL':
    num_channel = 3
    fc_input = 164864
    fc_output = 40
elif DATASET_NAME == 'MNIST':
    num_channel = 1
    fc_input = 12544
    fc_output = 10
elif DATASET_NAME == 'CIFAR':
    num_channel = 3
    fc_input = 16384
    fc_output = 10
# -----------------------

from code.stage_3_code.dataset_loader import Image_Dataset
from code.stage_3_code.method_CNN import Method_CNN
from code.stage_3_code.setting_CNN import CNN_Trainer
from code.stage_3_code.evaluate_CNN import Evaluate_CNN
from code.stage_3_code.result_CNN import Results_CNN
import numpy as np
import torch

if 1:
    #---- parameter section ----
    np.random.seed(2)
    torch.manual_seed(2)
    device = torch.device("cpu")
    if useGPU:
        if (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
            device = torch.device("mps")
            print("M1 GPU Detected. Using M1 GPU")
        elif (torch.cuda.is_available()):
            device = torch.device("cuda")
            print("nVidia GPU Detected. Using CUDA")
        else:
            device = torch.device("cpu")
            print("No compatible GPU Detected. Using CPU")
        print()

    #---- Object Initialization ----
    
    dataset = Image_Dataset(DATASET_NAME, '')
    dataset.dataset_source_file_name = DATASET_NAME
    dataset.dataset_source_folder_path = '../../data/stage_3_data/'
    
    method = Method_CNN('CNN', '', num_channel, fc_input, fc_output, device)
    
    result = Results_CNN('Saver', '')
    result.result_destination_folder_path = '../../result/stage_3_result/CNN_'
    result.result_destination_file_name = 'prediction_result-'+DATASET_NAME
    
    evaluate = Evaluate_CNN('accuracy', '')
    
    setting = CNN_Trainer(f'CNN {DATASET_NAME} trainer', '')
    
    #---- Running ----
    print('************ Start ************')
    setting.prepare(dataset, method, result, evaluate)
    setting.print_setup_summary()
    mean_score = setting.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print('CNN Accuracy: ' + str(mean_score))
    print('************ Finish ************')