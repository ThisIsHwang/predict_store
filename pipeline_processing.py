import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import csv
import time

pd.options.mode.chained_assignment = None

def circular_weight(distance):
    try:
        result = math.sqrt(1-distance**2)
    except:
        print("error! {}".format(distance))
    return math.sqrt(1-distance**2)

def gaussian_weight(distance):
    mu = 0
    sigma = 0.5
    return np.exp(-(distance - mu)**2 / (2 * sigma**2))

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

# Determine the distribution of store types within a 0.5 km radius
def get_distribution(row, data, radius=0.5):
    # Apply a mask to find stores within the specified radius
    mask = data.apply(lambda x: haversine(row['Longitude'], row['Latitude'], x['좌표정보(x)'], x['좌표정보(y)']) <= radius, axis=1)
    nearby_stores = data[mask]
    distribution = nearby_stores['업태구분명'].value_counts(normalize=True)
    return distribution.to_dict()

def get_weighted_distribution(row, data, max_radius=1.0):
    weight_within_05 = 2
    weight_within_1 = 1

    idx_list = row['nearby']
    data = data.iloc[idx_list]
    # drop '통신판매업'
    # mask = data['개방서비스명'] == '통신판매업'
    # mask = data['업태구분명'] == '통신판매업'
    # data = data[~mask]


    mask = data.apply(lambda x: haversine(row['Longitude'], row['Latitude'], x['좌표정보(x)'], x['좌표정보(y)']), axis=1)
    data['weight'] = mask.apply(lambda distance: gaussian_weight(distance = distance))

    nearby_stores = data[data['weight'] > 0]

    weighted_counts = nearby_stores.groupby('업태구분명')['weight'].sum()
    total_weight = weighted_counts.sum()
    weighted_distribution = (weighted_counts / total_weight).to_dict()

    return weighted_distribution

# Load the original dataset
original_data_path = 'dataset/updated_diningcode_output.csv'
df = pd.read_csv(original_data_path)

# Filter the DataFrame to include only valid entries
filtered_df = df[(df['score'] > 0) & (df['category'].notnull())]
filtered_df = filtered_df.reset_index(drop=True)

nearby_store = []
pipeline_data = 'dataset/process_pipeline1.csv'
pipeline_file = open(pipeline_data, 'r', newline='', encoding='utf-8')
# CSV 파일을 읽기 위한 reader 객체 생성
csv_reader = csv.reader(pipeline_file)
for row in csv_reader:
    temp = []
    for elem in row[0].split('/'):
        if elem == '':
            break
        temp.append(int(elem))
    nearby_store.append([temp])

# Save the filtered DataFrame to a new CSV file (optional)
# filtered_data_path = '/Users/hwangyun/PycharmProjects/pred_loc/dataset/filtered_diningcode_output.csv'
# filtered_df.to_csv(filtered_data_path, index=False)

# Load the filtered and merged dataset
merged_data_path = "dataset/final_merged_filtered_data.csv"
data = pd.read_csv(merged_data_path)

# Fill missing '업태구분명' values with '개방서비스명'
# data['업태구분명'] = data['업태구분명'].fillna(data['개방서비스명'])

# Use only the relevant columns for further processing
relevant_columns = ['좌표정보(x)', '좌표정보(y)', '업태구분명', '개방서비스명']
data = data[relevant_columns]

fnb_related = ['집단급식소', '제과점영업', '단란주점영업', '유흥주점영업', '관광식당', '관광유흥음식점업','외국인전용유흥음식점업','일반음식점','휴게음식점','대규모점포']

# '개방서비스명'이 fnb_related에 포함되지 않는 경우 '업태구분명'을 '개방서비스명'으로 업데이트
data.loc[~data['개방서비스명'].isin(fnb_related), '업태구분명'] = data['개방서비스명']

data = data[['좌표정보(x)', '좌표정보(y)', '업태구분명']]

# Calculate the distribution of nearby stores for each row
# Note: This may be slow for large datasets; consider parallel processing methods if necessary

# results = [get_distribution(row, data) for _, row in tqdm(filtered_df.iterrows(), total=filtered_df.shape[0])]
temp_df = pd.DataFrame(data = nearby_store, columns = ['nearby'])
nearby_data = pd.concat([filtered_df, temp_df],axis=1)
results = [get_weighted_distribution(rows, data) for _, rows in tqdm(nearby_data.iterrows(), total=nearby_data.shape[0])]

# Create a DataFrame from the distribution results and combine it with the filtered data
distribution_df = pd.DataFrame(results).fillna(0)
filtered_df = pd.concat([filtered_df.reset_index(drop=True), distribution_df], axis=1)

# Save the combined data with distribution information
intermediate_data_path = 'dataset/inter_diningcode_weighted_dropped_filtering.csv'
filtered_df.to_csv(intermediate_data_path, index=False)

# Load the data for the machine learning model
data = pd.read_csv(intermediate_data_path)

# Simplify the category labels by splitting on '>' and keeping the last element
data['category'] = data['category'].str.split('>').str[-1]

# Prepare the features (X) and labels (y) for the model
feature_columns = ['category', 'Title', 'Latitude', 'Longitude', 'Info Title', 'Title Place', 'hashes', 'score', 'userScore', 'heart', 'title', 'address', 'roadAddress', 'mapx', 'mapy', 'categories']
X = data.drop(columns=feature_columns)
y = data['category']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)