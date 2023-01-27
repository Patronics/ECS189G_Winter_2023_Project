
# -- compatibility layer --
import sys
import os
# print(sys.path)
# from the file, add the root directory to python path.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
# print(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)
# Then, we make sure the OS's cwd is at the local level to make it work
os.chdir(os.path.dirname(os.path.realpath(__file__)))
# -----------------------


from code.stage_2_code.Dataset_Loader import Dataset_Loader
from code.stage_2_code.Method_SVM import Method_SVM
from code.stage_2_code.Result_Saver import Result_Saver
from code.stage_2_code.Setting_KFold_CV import Setting_KFold_CV
from code.stage_2_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np

#---- Support Vector Machine script ----
if 1:
    #---- parameter section -------------------------------
    c = 1.0
    np.random.seed(2)
    #------------------------------------------------------

    # ---- objection initialization setction ---------------

    data_obj = Dataset_Loader('stage_2', '')
    data_obj.dataset_source_folder_path = '../../data/stage_2_data/'
    data_obj.dataset_source_file_name = 'train.csv'

    method_obj = Method_SVM('support vector machine', '')
    method_obj.c = c

    result_obj = Result_Saver('saver', '')

    result_obj.result_destination_folder_path = '../../result/stage_2_result/SVM_'
    result_obj.result_destination_file_name = 'prediction_result'

    #setting_obj = Setting_KFold_CV('k fold cross validation', '')
    setting_obj = Setting_Train_Test_Split('train test split', '')


    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    mean_score, std_score = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print('SVM Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
    print('************ Finish ************')
    # ------------------------------------------------------
    

    