import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Define the Haversine formula for calculating the great-circle distance between two points on Earth
def haversine(lon1, lat1, lon2, lat2):
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    # Radius of Earth in kilometers
    R = 6371
    return R * c

# Modified function to calculate weighted distribution of store types
def get_weighted_distribution(row, data, max_radius=1.0):
    # Define weights for different distance ranges
    weight_within_05 = 2  # Higher weight for stores within 0.5 km
    weight_within_1 = 1   # Lower weight for stores within 1 km

    # Apply a mask to find stores within the specified maximum radius and calculate weights
    mask = data.apply(lambda x: haversine(row['Longitude'], row['Latitude'], x['좌표정보(x)'], x['좌표정보(y)']), axis=1)
    data['weight'] = mask.apply(lambda distance: weight_within_05 if distance <= 0.5 else (weight_within_1 if distance <= max_radius else 0))

    # Filter out stores beyond the maximum radius
    nearby_stores = data[data['weight'] > 0]

    # Calculate the weighted distribution
    weighted_counts = nearby_stores.groupby('업태구분명')['weight'].sum()
    total_weight = weighted_counts.sum()
    weighted_distribution = (weighted_counts / total_weight).to_dict()

    return weighted_distribution

# Load the original dataset
original_data_path = '/Users/sanakang/Desktop/predict_store/dataset/updated_diningcode_output.csv'
df = pd.read_csv(original_data_path)

# Filter the DataFrame to include only valid entries
filtered_df = df[(df['score'] > 0) & (df['category'].notnull())]

# Save the filtered DataFrame to a new CSV file (optional)
# filtered_data_path = '/Users/hwangyun/PycharmProjects/pred_loc/dataset/filtered_diningcode_output.csv'
# filtered_df.to_csv(filtered_data_path, index=False)

# Load the filtered and merged dataset
merged_data_path = "/Users/sanakang/Desktop/predict_store/dataset/final_merged_filtered_data.csv"
data = pd.read_csv(merged_data_path, encoding='utf-8')

# # Fill missing '업태구분명' values with '개방서비스명'
# data['업태구분명'] = data['업태구분명'].fillna(data['개방서비스명'])

# # Use only the relevant columns for further processing
# relevant_columns = ['좌표정보(x)', '좌표정보(y)', '업태구분명']
# data = data[relevant_columns]

# Use only the relevant columns for further processing
relevant_columns = ['좌표정보(x)', '좌표정보(y)', '업태구분명', '개방서비스명']
data = data[relevant_columns]

# 집단급식소 = 병원, 사회복지시설, 어린이집, 산업체, 병원, etc.
fnb_related = ['집단급식소', '제과점영업', '단란주점영업', '유흥주점영업', '관광식당', '관광유흥음식점업','외국인전용유흥음식점업','일반음식점','휴게음식점','대규모점포']

# '개방서비스명'이 fnb_related에 포함되지 않는 경우 '업태구분명'을 '개방서비스명'으로 업데이트
data.loc[~data['개방서비스명'].isin(fnb_related), '업태구분명'] = data['개방서비스명']

data = data[['좌표정보(x)', '좌표정보(y)', '업태구분명']]

# Calculate the distribution of nearby stores for each row
# Note: This may be slow for large datasets; consider parallel processing methods if necessary
results = [get_weighted_distribution(row, data) for _, row in tqdm(filtered_df.iterrows(), total=filtered_df.shape[0])]

# Create a DataFrame from the distribution results and combine it with the filtered data
distribution_df = pd.DataFrame(results).fillna(0)
filtered_df = pd.concat([filtered_df.reset_index(drop=True), distribution_df], axis=1)

# Save the combined data with distribution information
intermediate_data_path = '/Users/sanakang/Desktop/predict_store/dataset/inter_diningcode_dropped.csv'
filtered_df.to_csv(intermediate_data_path, index=False)
filtered_df.to_csv('/Users/sanakang/Desktop/predict_store/dataset/testtesttest.csv', index=False)

# # Simplify the category labels by splitting on '>' and keeping the last element
# data['category'] = data['category'].str.split('>').str[-1]

# # Prepare the features (X) and labels (y) for the model
# feature_columns = ['category', 'Title', 'Latitude', 'Longitude', 'Info Title', 'Title Place', 'hashes', 'score', 'userScore', 'heart', 'title', 'address', 'roadAddress', 'mapx', 'mapy', 'categories']
# X = data.drop(columns=feature_columns)
# y = data['category']

# # Split the data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)