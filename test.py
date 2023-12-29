import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import csv
import time
from itertools import product
from process import Process
from process import HashProcess
from dataclasses import dataclass
import concurrent.futures
from functools import partial
import copy
from model import Model

# processing function
# return : cache_path_list
def processing(commercial_data_path, local_data_path, cache_index = ""):
    cache_path_list = []

    process = Process()

    # feature_param_list and weight_param_list is list of function
    # feature_param_list = [fill_all(), drop_unrelated(), ...]
    feature_param_list = process.get_feature_param()

    # weight_param_list = [no_weight_05(), gaussian()...]
    weight_param_list = process.get_weight_param()

    # distribution_param_list is list of attributes
    # distribution_param_list = ['drop_mail_order', 'fnb_related', ...]
    distribution_param_list = process.get_distribution_param()

    bool_combination = list(product([True, False], repeat=len(distribution_param_list)))

    for feature_param in feature_param_list:
        for weight_param in weight_param_list:
            for distribution_value_list in bool_combination:
                for i in range(len(distribution_param_list)):
                    process.distribution_param.set_value(distribution_param_list[i], distribution_value_list[i])
            
                process.feature_param = Process.Feature(feature_param)
                process.weight_param = Process.Weight(weight_param)

                cache = '{}_{}+{}+{}.csv'.format(cache_index, feature_param.__name__, weight_param.__name__, '+'.join(process.distribution_param.get_true_attributes()))
                result_data_path = 'cache/'+cache
                process.process_data(data_folder + commercial_data_path, data_folder + local_data_path, result_data_path)

                cache_path_list.append((cache, feature_param.__name__, weight_param.__name__, '+'.join(process.distribution_param.get_true_attributes())))
                print('{} is stored in cache file'.format(cache))

    return cache_path_list

# processing function
# return : cache_path_list
def hash_processing(commercial_data_path, local_data_path, cache_index = ""):
    cache_path_list = []

    process = HashProcess()

    # feature_param_list and weight_param_list is list of function
    # feature_param_list = [fill_all(), drop_unrelated(), ...]
    feature_param_list = process.get_feature_param()

    # weight_param_list = [no_weight_05(), gaussian()...]
    weight_param_list = process.get_weight_param()

    for feature_param in feature_param_list:
        for weight_param in weight_param_list:
            process.feature_param = Process.Feature(feature_param)
            process.weight_param = Process.Weight(weight_param)

            cache = '{}_hash_{}+{}.csv'.format(cache_index, feature_param.__name__, weight_param.__name__)
            result_data_path = 'cache/'+cache
            process.process_data(data_folder + commercial_data_path, data_folder + local_data_path, result_data_path)

            cache_path_list.append((cache, feature_param.__name__, weight_param.__name__, None))
            print('{} is stored in cache file'.format(cache))

    return cache_path_list

def load_data(processed_data_paths, hash_path_list):
    def load_and_drop(path):
        data = pd.read_csv(path)
        columns_to_drop = ['categories', 'Info Title', 'Title Place', 'hashes', 'score', 'userScore', 'heart', 'title',
                        'address', 'roadAddress', 'mapx', 'mapy']
        data.drop(columns=columns_to_drop, inplace=True)

        return data
    
    weight_idx = processed_data_paths[0][2]
    for i in range(len(hash_path_list)):
        if hash_path_list[i][0][2] == weight_idx:
            hash_paths = hash_path_list[i]
            break
        
    data_folder = "cache/"

    total_data = []
    for i in range(len(processed_data_paths)):
        processed_data = load_and_drop(data_folder + processed_data_paths[i][0])
        hash_data = load_and_drop(data_folder + hash_paths[i][0])
        merge_data=pd.merge(processed_data, hash_data,
                                   on=['Title', 'Latitude', 'Longitude', 'category'], how='inner')
        # print(merge_data)
        total_data.append(pd.merge(processed_data, hash_data,
                                   on=['Title', 'Latitude', 'Longitude', 'category'], how='inner'))
        # print(total_data)
        total_data[i] = total_data[i].drop_duplicates(subset=['Title', 'Latitude', 'Longitude', 'category'], inplace=False)
        
    data = pd.concat(total_data, ignore_index=True)

    # Final preprocessing steps (if any)
    data['category'] = data['category'].str.split('>').str[-1]

    return data

def get_param_name(processed_data_paths):
    return '{' + '}+{'.join([processed_data_paths[0][1], processed_data_paths[0][2], processed_data_paths[0][3]]) + '}'

if __name__ == '__main__':

    previous_best = "~~~"

    area_idx_list = [
        "youngdeungpo",
        "jongro"
    ]

    data_folder = "data/"
    commercial_data_path_list = [
        "updated_diningcode_youngdeungpo_1124.csv",
        "updated_diningcode_jongro_1124.csv"
    ]
    local_data_path_list = [
        "final_merged_filtered_youngdeungpo_data.csv",
        "final_merged_filtered_jongro_data.csv"
    ]

    processed_data_path_list = []
    for i in range(len(area_idx_list)):
        commercial_data_path = commercial_data_path_list[i]
        local_data_path = local_data_path_list[i]
        area = area_idx_list[i]
        processed_data_path_list.append(
            processing(commercial_data_path=commercial_data_path,
                       local_data_path=local_data_path,
                       cache_index=area))
    # 2차원 리스트 뒤집기
    processed_data_path_list = list(map(list, zip(*processed_data_path_list)))
    
    hash_path_list = []
    for i in range(len(area_idx_list)):
        commercial_data_path = commercial_data_path_list[i]
        area = area_idx_list[i]
        hash_path_list.append(
            hash_processing(commercial_data_path=commercial_data_path,
                            local_data_path=commercial_data_path,
                            cache_index=area))
    # 2차원 리스트 뒤집기
    hash_path_list = list(map(list, zip(*hash_path_list)))
    

    max_accuracy = 0
    current_best = ""
    for processed_data_paths in processed_data_path_list:
        data = load_data(processed_data_paths, hash_path_list)
        
        start_time = time.time()
        param_name = get_param_name(processed_data_paths)
        print("Train {} ...".format(param_name))

        model = Model()
        X, y = model.preprocess_data(data)
        mean_scores = model.train_and_evaluate(X, y)

        elapsed_time = time.time()-start_time
        print(f"*** '{param_name}' test complete *** <Elpased time> : {elapsed_time}")
        print(f"Accuracy: {mean_scores[0]}")
        print(f"F1 Score (Micro): {mean_scores[1]}")
        print(f"F1 Score (Macro): {mean_scores[2]}")
        print(f"F1 Score (Weighted): {mean_scores[3]}")
        print(f"***********************************************************")

        if max_accuracy < mean_scores[0]:
            max_accuracy = mean_scores[0]
            current_best = param_name

    print(f"Previous Best parameter : {previous_best}")
    print(f"Current Best parameter : {current_best}")

    
    
