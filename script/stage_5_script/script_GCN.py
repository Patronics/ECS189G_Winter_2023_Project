# -- compatibility layer --
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

sys.path.insert(0, ROOT_DIR)
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# -----------------------

# -- configuration variables -- #
useGPU = True
#  uncomment or edit to whichever dataset you'd like to run the network on by default
DATASET_NAME = 'Cora' #choose from 'Cora', 'Citeseer', or 'Pubmed'
MODEL_TYPE = "GCN"



if (len(sys.argv)==2):
    DATASET_NAME = sys.argv[1]
    print("detected command line argument '"+sys.argv[1]+"', attempting to use that dataset")
    
if (len(sys.argv)==3):
    DATASET_NAME = sys.argv[1]
    print("detected command line argument '"+sys.argv[1]+"', attempting to use that dataset")
    model_type = sys.argv[2]
    print("detected command line argument '"+sys.argv[2]+"', attempting to use that model type")
#else:
#    print("invalid dataset name provided, please choose from [ORL, MNIST, CIFAR]")
#    sys.exit(1)
# -----------------------

from codes.stage_5_code.Dataset_Loader_Node_Classification import Dataset_Loader
from codes.stage_5_code.method_GCN import Method_GCN_Class
from codes.stage_5_code.setting_GCN import GCN_Trainer
from codes.stage_5_code.evaluate_GCN import Evaluate_GCN
from codes.stage_5_code.result_GCN import Results_GCN
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

    dataset_loader = Dataset_Loader(DATASET_NAME, '')
    dataset = dataset_loader.load()
    method = Method_GCN_Class('GCN', '', in_features=dataset['graph']['X'], out_features=dataset['graph']['Y'],rnn_model=model_type ,mDevice=device)

    dataset.dataset_source_file_name = DATASET_NAME #not used
    dataset.dataset_source_folder_path = '../../data/'
    

    
    result = Results_RNN('Saver', '')
    result.result_destination_folder_path = '../../result/stage_5_result/RNN_'
    result.result_destination_file_name = 'prediction_result-'+DATASET_NAME
    
    evaluate = Evaluate_RNN('accuracy', '')
    
    setting = RNN_Trainer(f'RNN {DATASET_NAME} trainer', '')
    
    #---- Running ----
    print('************ Start ************')
    setting.prepare(dataset, method, result, evaluate)
    setting.print_setup_summary()
    mean_score = setting.load_run_save_evaluate(DATASET_NAME)
    print('************ Overall Performance ************')
    print('RNN Accuracy: ' + str(mean_score))
    print('************ Finish ************')