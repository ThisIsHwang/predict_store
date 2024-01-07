import numpy as np
import pandas as pd

from model import ModelForPredictStoreType, ModelForPredictStoreHashes
from utils import load_data
from process import Process, HashProcess
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

class RecommendationService:
    def __init__(self):
        self.model_store_type = ModelForPredictStoreType()
        self.model_store_hashes = ModelForPredictStoreHashes()
        self.process = Process()
        self.process.feature_param = Process.Feature(param=Process.Feature.drop_unrelated)
        self.process.weight_param = Process.Weight(param=Process.Weight.gaussian_weight)
        self.process.distribution_param = Process.Distribution(fnb_related=True, drop_mail_order=True)

        self.hash_process = HashProcess()
        self.hash_process.feature_param = HashProcess.Feature()
        self.hash_process.weight_param = HashProcess.Weight(param=HashProcess.Weight.gaussian_weight)

        self.data_for_type, self.data_for_hash = load_data()

        # Load and preprocess the data
        self.X, self.y = self.model_store_type.preprocess_data(self.data_for_type)
        self.X_hashes, self.y_hashes = self.model_store_hashes.preprocess_data(self.data_for_hash)

        # Train models
        try:
            self.model_store_type.load_model()
        except Exception as e:
            print(e)
            self.model_store_type.train(self.X, self.y)
            self.model_store_type.save_model()

        try:
            self.model_store_hashes.load_model()
        except Exception as e:
            print(e)
            self.model_store_hashes.train(self.X_hashes, self.y_hashes)
            self.model_store_hashes.save_model()

    def _preprocess_location(self, latitude, longitude, location_type='gangnam'):
        # Placeholder for preprocessing logic
        # This would involve creating a data frame similar to the training data but for the input location
        row = {"Latitude": latitude, "Longitude": longitude}
        
        DATA_PATH = 'data/'
        
        commercial_data_path = DATA_PATH+f'updated_diningcode_{location_type}_20241.csv'
        local_data_path = DATA_PATH+f'final_merged_filtered_{location_type}_data_20241.csv'
        result_data_path = DATA_PATH+f'inter_diningcode_{location_type}_dropped_20241.csv'
        hash_commercial_data_path = DATA_PATH+f'updated_diningcode_{location_type}_20241.csv'
        hash_local_data_path = DATA_PATH+f'updated_diningcode_{location_type}_20241.csv'
        hash_result_data_path = DATA_PATH+f'inter_hashes_{location_type}_dropped_20241.csv'
        preprocessed_type_data = self.process.process_row(commercial_data_path, local_data_path, result_data_path, row)
        preprocessed_hash_data = self.hash_process.process_row(hash_commercial_data_path, hash_local_data_path, hash_result_data_path, row)
        return preprocessed_type_data, preprocessed_hash_data

    def get_top_5_recommendations(self, latitude, longitude):
        # Preprocess location data
        #youngdeungpo = (merged_df['좌표정보(y)'] >= 37.4789987952223) & (merged_df['좌표정보(y)'] <= 37.5548515573392) & \
            #(merged_df['좌표정보(x)'] >= 126.873119215493) & (merged_df['좌표정보(x)'] <= 126.950796763912)
        #jongro = (merged_df['좌표정보(y)'] >= 37.56336469999967) & (merged_df['좌표정보(y)'] <= 37.64290239999954) & \
        #(merged_df['좌표정보(x)'] >= 126.93943557353703) & (merged_df['좌표정보(x)'] <= 127.02377357353774)
        #gangnam = (merged_df['좌표정보(y)'] >= 37.44908183324762) & (merged_df['좌표정보(y)'] <= 37.53075444273274) & \
        #          (merged_df['좌표정보(x)'] >= 126.97612011196603) & (merged_df['좌표정보(x)'] <= 127.08357120788159)
        if (latitude >= 37.4789987952223) & (latitude <= 37.5548515573392) & \
            (longitude >= 126.873119215493) & (longitude <= 126.950796763912):
            location_type = 'youngdeungpo'
        elif (latitude >= 37.56336469999967) & (latitude <= 37.64290239999954) & \
            (longitude >= 126.93943557353703) & (longitude <= 127.02377357353774):
            location_type = 'jongro'
        elif (latitude >= 37.44908183324762) & (latitude <= 37.53075444273274) & \
            (longitude >= 126.97612011196603) & (longitude <= 127.08357120788159):
            location_type = 'gangnam'
        else:
            raise Exception("Location is not available.")
        preprocessed_type_data, preprocessed_hash_data = self._preprocess_location(latitude, longitude)

        # Convert preprocessed data to DataFrame
        preprocessed_type_df = pd.DataFrame([preprocessed_type_data])
        preprocessed_hash_df = pd.DataFrame([preprocessed_hash_data])

        # Ensure the column order is the same as in 'given_x'
        preprocessed_type_df = self._match_column_order(self.X, preprocessed_type_df)
        preprocessed_hash_df = self._match_column_order(self.X_hashes, preprocessed_hash_df)

        # Combine both DataFrames
        combined_df = self._combine_dataframes(preprocessed_type_df, preprocessed_hash_df)
        combined_df = self.model_store_type.scaler.transform(combined_df)
        # Predict store types and hashes
        store_types_pred = self.model_store_type.xgb_classifier.predict_proba(combined_df)
        top_5_types = self._get_top_5_store_types(store_types_pred)

        combined_df = self._align_and_combine_dataframes(preprocessed_hash_df, preprocessed_type_df)
        # mark combined_df as whether_{top_5_types}
        # store_hashes_pred = []
        # for top_5_type in top_5_types:
        #     copied_combined_df = combined_df.copy(deep=True)
        #     copied_combined_df['category_' + top_5_type] = 1
        #     copied_combined_df = self.model_store_hashes.scaler.transform(copied_combined_df)
        #     temp = self.model_store_hashes.classifier.predict(copied_combined_df)
        #     store_hashes_pred.append(temp)
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self._predict_hashes, combined_df, top_5_type) for top_5_type in top_5_types]
            store_hashes_pred = [future.result() for future in futures]

        types_hashes = self._get_hashes(store_hashes_pred)

        return [{"type":t, 'hashes': h} for t, h in zip(top_5_types, types_hashes)]


    def _match_column_order(self, reference_df, target_df):
        # Identify columns in reference_df that are not in target_df
        missing_cols = [col for col in reference_df.columns if col not in target_df.columns]

        # If there are missing columns, create a new DataFrame with these columns filled with default values
        if missing_cols:
            missing_df = pd.DataFrame(0, index=target_df.index, columns=missing_cols)
            target_df = pd.concat([target_df, missing_df], axis=1)

        # Reorder columns of target_df to match reference_df
        return target_df[reference_df.columns]

    def _get_top_5_store_types(self, predictions_proba):
        # Assuming 'predictions' is an array of predicted labels
        top_5_indices = np.argsort(np.max(predictions_proba, axis=0))[::-1][:5]
        top_5_types = self.model_store_type.label_encoder.inverse_transform(top_5_indices)
        return top_5_types.tolist()

    def _process_prediction_row(self, prediction_row, hash_column_names):
        # Process a single prediction row to extract hashes
        return {col[8:].strip() for col, val in zip(hash_column_names, prediction_row) if val == 1}

    def _predict_hashes(self, combined_df, top_5_type):
        copied_combined_df = combined_df.copy(deep=True)
        copied_combined_df['category_' + top_5_type] = 1
        copied_combined_df = self.model_store_hashes.scaler.transform(copied_combined_df)
        return self.model_store_hashes.classifier.predict(copied_combined_df)

    def _get_hashes(self, predictions_list):
        hash_column_names = self.model_store_hashes.y_columns

        all_hash_strings = []

        # Parallel processing
        with ThreadPoolExecutor() as executor:
            for predictions in predictions_list:
                futures = [executor.submit(self._process_prediction_row, row, hash_column_names) for row in predictions]
                hashes_for_each_pred = [future.result() for future in futures]
                all_hash_strings.extend(hashes_for_each_pred)

        # Return list of sets of unique hashes
        return all_hash_strings

    def _combine_dataframes(self, df1, df2):
        common_columns = df1.columns.intersection(df2.columns)
        combined_df = df1[common_columns] + df2[common_columns]  # Example operation: addition
        return combined_df

    def _align_and_combine_dataframes(self, df1, df2):
        # Create a union of columns from both DataFrames
        larger_df_columns = df1.columns if len(df1.columns) > len(df2.columns) else df2.columns

        # Align both DataFrames to the same set of columns
        aligned_df1 = df1.reindex(columns=larger_df_columns, fill_value=0)
        aligned_df2 = df2.reindex(columns=larger_df_columns, fill_value=0)

        # Perform your desired operation - for example, addition
        combined_df = aligned_df1 + aligned_df2

        return combined_df

# Example usage
if __name__ == '__main__':
    service = RecommendationService()
    '''
    37.5027525	127.0473401
    37.5054451	127.0529798
    37.5044822	127.042922
    37.5046128	127.0545895
    37.5042782	127.0544302
    37.5042878	127.0454573
    37.5025508	127.0521345
    37.5054705	127.0425347
    37.5031735	127.0509222
    37.505062	127.0465238
    37.511777	127.055942
    37.5005596	127.0425154
    37.5042377	127.0543081
    37.5047832	127.0532504
    37.5031049	127.0479738
    37.5027312	127.0549182
    37.5057604	127.0540601
    37.502736	127.0520046
    37.5058388	127.0561271
    37.506801	127.053439
    37.5115064	127.044629
    37.5102986	127.0541293
    37.506751	127.05398
    37.5053965	127.0430921
    37.5026462	127.0508956
    37.5048921	127.0450173
    37.5025808	127.0527293
    37.5005158	127.0533101
    37.508373	127.0547157
    37.5011291	127.052184
    37.5062299	127.0478646
    37.5080738	127.0556549
    37.504079	127.0558008
    37.5012206	127.0553487
    37.5013724	127.0529503
    37.5110739	127.0557318
    37.504924	127.046745
    37.5036853	127.054471
    37.5040472	127.0522945
    37.5044597	127.0521332
    37.5025714	127.0520978
    37.5046521	127.0442405
    37.5032716	127.0489161
    37.5034423	127.0476061
    37.5055786	127.0533872
    37.507999	127.0551621
    37.5033933	127.0427641
    37.5029807	127.046554
    37.5028977	127.0484324
    '''
    #make list for lat, lon
    lat_list = [37.5032108, 37.5037241, 37.5027525, 37.5054451, 37.5044822, 37.5046128, 37.5042782, 37.5042878, 37.5025508, 37.5054705, 37.5031735, 37.505062, 37.511777, 37.5005596, 37.5042377, 37.5047832, 37.5031049, 37.5027312, 37.5057604, 37.502736, 37.5058388, 37.506801, 37.5115064, 37.5102986, 37.506751, 37.5053965, 37.5026462, 37.5048921, 37.5025808, 37.5005158, 37.508373, 37.5011291, 37.5062299, 37.5080738, 37.504079, 37.5012206, 37.5013724, 37.5110739, 37.504924, 37.5036853, 37.5040472, 37.5044597, 37.5025714, 37.5046521, 37.5032716, 37.5034423, 37.5055786, 37.507999, 37.5033933, 37.5029807, 37.5028977]
    lon_list = [127.0520516, 127.0546461, 127.0473401, 127.0529798, 127.042922, 127.0545895, 127.0544302, 127.0454573, 127.0521345, 127.0425347, 127.0509222, 127.0465238, 127.055942, 127.0425154, 127.0543081, 127.0532504, 127.0479738, 127.0549182, 127.0540601, 127.0520046, 127.0561271, 127.053439, 127.044629, 127.0541293, 127.05398, 127.0430921, 127.0508956, 127.0450173, 127.0527293, 127.0533101, 127.0547157, 127.052184, 127.0478646, 127.0556549, 127.0558008, 127.0553487, 127.0529503, 127.0557318, 127.046745, 127.054471, 127.0522945, 127.0521332, 127.0520978, 127.0442405, 127.0489161, 127.0476061, 127.0533872, 127.0551621, 127.0427641, 127.046554, 127.0484324]
    for lat, lon in zip(lat_list, lon_list):
        top_recommendations = service.get_top_5_recommendations(lat, lon)
        print(top_recommendations)

