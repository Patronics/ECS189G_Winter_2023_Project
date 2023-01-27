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


from code.stage_2_code.Result_Loader import Result_Loader

if 1:
    result_obj = Result_Loader('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_2_result/SVM_'
>>>>>>> Stashed changes
    result_obj.result_destination_file_name = 'prediction_result'

    for fold_count in [1, 2, 3, None]:
        result_obj.fold_count = fold_count
        result_obj.load()
        print('Fold:', fold_count, ', Result:', result_obj.data)