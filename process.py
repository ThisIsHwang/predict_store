import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import csv
import time
import concurrent.futures
from functools import partial
import copy

pd.options.mode.chained_assignment = None
    
class Process:
    class Feature:
        def fill_all(data):
            # Fill missing '업태구분명' values with '개방서비스명'
            data['업태구분명'] = data['업태구분명'].fillna(data['개방서비스명'])
            
            relevant_columns = ['좌표정보(x)', '좌표정보(y)', '업태구분명', '개방서비스명']
            data = data[relevant_columns]
            return data
        
        def drop_unrelated(data):
            fnb_related = ['집단급식소', '제과점영업', '단란주점영업', '유흥주점영업', '관광식당', '관광유흥음식점업','외국인전용유흥음식점업','일반음식점','휴게음식점','대규모점포']

            # '개방서비스명'이 fnb_related에 포함되지 않는 경우 '업태구분명'을 '개방서비스명'으로 업데이트
            data.loc[~data['개방서비스명'].isin(fnb_related), '업태구분명'] = data['개방서비스명']
            
            relevant_columns = ['좌표정보(x)', '좌표정보(y)', '업태구분명', '개방서비스명']
            data = data[relevant_columns]
            return data

        def __init__(self, param=fill_all):
            self.param = param

        def modify_feature(self, data):
            return self.param(data)
        
        def get_param(self):
            return [Process.Feature.fill_all,
                    Process.Feature.drop_unrelated]
            
    class Weight:
        def circular_weight(distances):
            clipped_distances = np.clip(distances, 0, 1)
            return np.sqrt(1 - clipped_distances**2)

        def gaussian_weight(distances):
            mask = distances <= 1
            mu = 0
            sigma = 0.5
            return np.where(mask, 2 * np.exp(-(distances - mu)**2 / (2 * sigma**2)), 0)

        def no_weight_05(distances):
            return np.where(distances <= 0.5, 1, 0)

        def double_weight_05(distances):
            return np.where(distances <= 0.5, 1, 2)
        
        def __init__(self, param = no_weight_05):
            self.param = param
        
        def calculate_weight(self, distances):
            return self.param(distances)
        
        def get_param(self):
            return[Process.Weight.circular_weight,
                   Process.Weight.gaussian_weight,
                   Process.Weight.no_weight_05,
                   Process.Weight.double_weight_05]

    class Distribution:

        def __init__(self, drop_mail_order = False, fnb_related = False):
            self.drop_mail_order = drop_mail_order
            self.fnb_related = fnb_related

        def get_true_attributes(self):
            return [attr for attr, value in vars(self).items() if value]
        
        def get_param_name(self):
            attribute_names = [name for name in vars(self) if not name.startswith("__")]
            return attribute_names
        
        def set_value(self, attribute_name, value):
            # This method sets the value of the specified attribute
            if hasattr(self, attribute_name):
                setattr(self, attribute_name, value)
            else:
                raise AttributeError(f"Attribute '{attribute_name}' not found in the Distribution class")

    def __init__(self):
        self.feature_param = Process.Feature()
        self.weight_param = Process.Weight()
        self.distribution_param = Process.Distribution()

    def __vectorized_haversine(self, lat1, lon1, lat2_array, lon2_array):
        # Vectorized Haversine formula
        lon1, lat1, lon2_array, lat2_array = np.radians(lon1), np.radians(lat1), np.radians(lon2_array), np.radians(lat2_array)
        dlon = lon2_array - lon1
        dlat = lat2_array - lat1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2_array) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        R = 6371  # Radius of Earth in kilometers
        return R * c
    
    def get_weighted_distribution(self, row_tuple, data, radius=1.0):
        _, row = row_tuple

        if self.distribution_param.drop_mail_order:
            # drop '통신판매업'
            mask = data['업태구분명'] == '통신판매업'
            data = data[~mask]
        
        # Vectorized  havesine function
        distances = self.__vectorized_haversine(row['Latitude'], row['Longitude'], data['좌표정보(y)'].values, data['좌표정보(x)'].values)
        mask = distances < 1.0
        data['weight'] = np.where(mask, self.weight_param.calculate_weight(distances=distances), 0)

        nearby_stores = data[data['weight'] > 0]

        if self.distribution_param.fnb_related:
            fnb_related = ['집단급식소', '제과점영업', '단란주점영업', '유흥주점영업', '관광식당', '관광유흥음식점업','외국인전용유흥음식점업','일반음식점','휴게음식점','대규모점포']
            nearby_stores.loc[nearby_stores['개방서비스명'].isin(fnb_related), 'weight'] = nearby_stores['weight'] * 2

        weighted_counts = nearby_stores.groupby('업태구분명')['weight'].sum()
        total_weight = weighted_counts.sum()
        weighted_distribution = (weighted_counts / total_weight).to_dict()

        return weighted_distribution

    def calculate_weighted_distribution(self, row, data):
        return self.get_weighted_distribution(row, data)

    def __read_data(self, commercial_data_path, local_data_path):
        df = pd.read_csv(commercial_data_path, low_memory=False)
        filtered_df = df[(df['score'] > 0) & (df['category'].notnull())]
        filtered_df = filtered_df.reset_index(drop=True)

        chunk_size = 10000
        data_iter = pd.read_csv(local_data_path, low_memory=False, chunksize=chunk_size)

        data_list = []
        for chunk in data_iter:
            data_chunk = self.feature_param.modify_feature(chunk)
            data_list.append(data_chunk)
        data = pd.concat(data_list, ignore_index=True)

        return filtered_df, data

    def process_data(self, commercial_data_path, local_data_path, result_data_path):
        filtered_df, data = self.__read_data(commercial_data_path=commercial_data_path,
                                           local_data_path=local_data_path)

        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Define a partial function with fixed parameters (data, weight, feature_parameter)
            partial_calculation = partial(self.calculate_weighted_distribution, data=data)
            
            # Map the function to the rows in parallel
            results = list(tqdm(executor.map(partial_calculation, filtered_df.iterrows()), total=filtered_df.shape[0]))
        
        distribution_df = pd.DataFrame(results).fillna(0)
        combined_df = pd.concat([filtered_df.reset_index(drop=True), distribution_df], axis=1)

        combined_df.to_csv(result_data_path, index = False)

        print('intermediate data is stored in {}'.format(result_data_path))
    
    def get_weight_param(self):
        return self.weight_param.get_param()
    
    def get_feature_param(self):
        return self.feature_param.get_param()
    
    def get_distribution_param(self):
        return self.distribution_param.get_param_name()
    
class HashProcess:
    class Feature:
        def no_feature(data):
            relevant_columns = ['Latitude', 'Longitude', 'hashes']
            data = data[relevant_columns]

            # split by hash and make 'hash' column
            data['hash'] = data['hashes'].apply(lambda x: [item.strip() for item in x.split(',') if item.strip()])
            data = data.explode('hash').reset_index(drop=True)
            data = data.drop('hashes', axis = 1)

            # remove NAN and empty value
            data = data.dropna(subset=['hash'])
            data = data[data['hash'].apply(len) > 0]

            return data

        def __init__(self, param = no_feature):
            self.param = param

        def modify_feature(self, data):
            return self.param(data)
        
        def get_param(self):
            return [HashProcess.Feature.no_feature]
            
    class Weight:
        def circular_weight(distances):
            clipped_distances = np.clip(distances, 0, 1)
            return np.sqrt(1 - clipped_distances**2)

        def gaussian_weight(distances):
            mask = distances <= 1
            mu = 0
            sigma = 0.5
            return np.where(mask, 2 * np.exp(-(distances - mu)**2 / (2 * sigma**2)), 0)

        def no_weight_05(distances):
            return np.where(distances <= 0.5, 1, 0)

        def double_weight_05(distances):
            return np.where(distances <= 0.5, 1, 2)
        
        def __init__(self, param = no_weight_05):
            self.param = param
        
        def calculate_weight(self, distances):
            return self.param(distances)
        
        def get_param(self):
            return[HashProcess.Weight.circular_weight,
                   HashProcess.Weight.gaussian_weight,
                   HashProcess.Weight.no_weight_05,
                   HashProcess.Weight.double_weight_05]

    # Not used for this time
    # If you want to use this, remove annotation and then fill todo!()
    # All code which relavance with 'class Distribution', I write annotation 'Distribution todo!()'

    # class Distribution:

    #     def __init__(self, todo!()=False):
    #         self.todo!() = todo!()

    #     def get_true_attributes(self):
    #         return [attr for attr, value in vars(self).items() if value]
        
    #     def get_param_name(self):
    #         attribute_names = [name for name in vars(self) if not name.startswith("__")]
    #         return attribute_names
        
    #     def set_value(self, attribute_name, value):
    #         # This method sets the value of the specified attribute
    #         if hasattr(self, attribute_name):
    #             setattr(self, attribute_name, value)
    #         else:
    #             raise AttributeError(f"Attribute '{attribute_name}' not found in the Distribution class")

    def __init__(self):
        self.feature_param = HashProcess.Feature()
        self.weight_param = HashProcess.Weight()
        # Distribution todo!()
        # self.distribution_param = HashProcess.Distribution()

    def __vectorized_haversine(self, lat1, lon1, lat2_array, lon2_array):
        # Vectorized Haversine formula
        lon1, lat1, lon2_array, lat2_array = np.radians(lon1), np.radians(lat1), np.radians(lon2_array), np.radians(lat2_array)
        dlon = lon2_array - lon1
        dlat = lat2_array - lat1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2_array) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        R = 6371  # Radius of Earth in kilometers
        return R * c
    
    def get_weighted_distribution(self, row_tuple, data, radius=1.0):
        _, row = row_tuple
        
        # Vectorized  havesine function
        distances = self.__vectorized_haversine(row['Latitude'], row['Longitude'], data['Latitude'].values, data['Longitude'].values)
        mask = distances < 1.0
        data['weight'] = np.where(mask, self.weight_param.calculate_weight(distances=distances), 0)

        nearby_stores = data[data['weight'] > 0]

        weighted_counts = nearby_stores.groupby('hash')['weight'].sum()
        total_weight = weighted_counts.sum()
        weighted_distribution = (weighted_counts / total_weight).to_dict()

        return weighted_distribution

    def calculate_weighted_distribution(self, row, data):
        return self.get_weighted_distribution(row, data)

    def __read_data(self, commercial_data_path, local_data_path):
        df = pd.read_csv(commercial_data_path, low_memory=False)
        filtered_df = df[(df['score'] > 0) & (df['category'].notnull())]
        filtered_df = filtered_df.reset_index(drop=True)

        chunk_size = 10000
        data_iter = pd.read_csv(local_data_path, low_memory=False, chunksize=chunk_size)

        data_list = []
        for chunk in data_iter:
            data_chunk = self.feature_param.modify_feature(chunk)
            data_list.append(data_chunk)
        data = pd.concat(data_list, ignore_index=True)

        return filtered_df, data

    def process_data(self, commercial_data_path, local_data_path, result_data_path):
        filtered_df, data = self.__read_data(commercial_data_path=commercial_data_path,
                                           local_data_path=local_data_path)

        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Define a partial function with fixed parameters (data, weight, feature_parameter)
            partial_calculation = partial(self.calculate_weighted_distribution, data=data)
            
            # Map the function to the rows in parallel
            results = list(tqdm(executor.map(partial_calculation, filtered_df.iterrows()), total=filtered_df.shape[0]))
        
        distribution_df = pd.DataFrame(results).fillna(0)
        combined_df = pd.concat([filtered_df.reset_index(drop=True), distribution_df], axis=1)

        combined_df.to_csv(result_data_path, index = False)

        print('intermediate data is stored in {}'.format(result_data_path))
    
    def get_weight_param(self):
        return self.weight_param.get_param()
    
    def get_feature_param(self):
        return self.feature_param.get_param()
    
    def get_distribution_param(self):
        return self.distribution_param.get_param_name()

def main():

    process = Process()

    process.feature_param = Process.Feature(param=Process.Feature.drop_unrelated)
    process.weight_param = Process.Weight(param=Process.Weight.gaussian_weight)
    process.distribution_param = Process.Distribution(fnb_related=True, drop_mail_order=True)
    
    # Above code is same to this code
    # process.feature_param = Process.Feature(Process.Feature.fill_all)
    # process.weight_param = Process.Weight(Process.Weight.no_weight_05)
    # process.distribution_param = Process.Distribution(
    #     drop_mail_order=False,
    #     fnb_related=False
    # )

    commercial_data_path = 'data/updated_diningcode_gangnam_20241.csv'
    local_data_path = 'data/final_merged_filtered_gangnam_data_20241.csv'
    result_data_path = 'data/inter_diningcode_gangnam_dropped_20241.csv'
    process.process_data(commercial_data_path, local_data_path, result_data_path)
    #process.process_data(commercial_data_path, local_data_path, result_data_path)


    hash_process = HashProcess()

    hash_process.feature_param = HashProcess.Feature()
    hash_process.weight_param = HashProcess.Weight(param=HashProcess.Weight.gaussian_weight)
    # Distribution todo!()
    # hash_process.distribution_param = HashProcess.Distribution()
    
    # Above code is same to this code
    # hash_process.feature_param = HashProcess.Feature(HashProcess.Feature.no_feature)
    # hash_process.weight_param = HashProcess.Weight(HashProcess.Weight.no_weight_05)
    # hash_process.distribution_param = Process.Distribution(
    #     self.todo!() = False
    # )
    
    
    hash_commercial_data_path = 'data/updated_diningcode_gangnam_20241.csv'
    hash_local_data_path = 'data/updated_diningcode_gangnam_20241.csv'
    hash_result_data_path = 'data/inter_hashes_gangnam_dropped_20241.csv'
    hash_process.process_data(hash_commercial_data_path, hash_local_data_path, hash_result_data_path)

if __name__ == '__main__':
    main()
