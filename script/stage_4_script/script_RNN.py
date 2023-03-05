# -- compatibility layer --
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

sys.path.insert(0, ROOT_DIR)
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# -----------------------

# -- configuration variables -- #
useGPU = True
model_type = "lstm"

#    uncomment whichever dataset you'd like to run the network on by default
# DATASET_NAME = 'GEN'
DATASET_NAME = 'Class'

# -----------------------

from codes.stage_4_code.dataset_RNN import Text_Dataset
from codes.stage_4_code.method_RNN import Method_RNN_Class, Method_RNN_Gen
from codes.stage_4_code.setting_RNN import RNN_Trainer
from codes.stage_4_code.evaluate_RNN import Evaluate_RNN
from codes.stage_4_code.result_RNN import Results_RNN
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

    if DATASET_NAME == "Class":
        method = Method_RNN_Class('RNN', '', vocab_size=2196016, rnn_model=model_type ,mDevice=device)
        dataset = Text_Dataset(DATASET_NAME, '')
    else:
        method = Method_RNN_Gen('RNN', '', vocab_size=2196016, rnn_model=model_type ,mDevice=device)
        dataset = Text_Dataset(DATASET_NAME, '')

    dataset.dataset_source_file_name = DATASET_NAME
    dataset.dataset_source_folder_path = '../../data/stage_4_data/'
    

    
    result = Results_RNN('Saver', '')
    result.result_destination_folder_path = '../../result/stage_4_result/RNN_'
    result.result_destination_file_name = 'prediction_result-'+DATASET_NAME
    
    evaluate = Evaluate_RNN('accuracy', '')
    
    setting = RNN_Trainer(f'RNN {DATASET_NAME} trainer', '')
    
    #---- Running ----
    print('************ Start ************')
    setting.prepare(dataset, method, result, evaluate)
    setting.print_setup_summary()
    mean_score = setting.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print('RNN Accuracy: ' + str(mean_score))
    print('************ Finish ************')