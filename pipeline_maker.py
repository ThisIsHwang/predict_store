import pandas as pd
import numpy as np
import math
from tqdm import tqdm
import csv

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

def get_border(lon, lat):
    # diff = 0.0001
    # while haversine(lon, lat, lon+diff, lat) <= 1:
    #     diff += 0.0001
    diff = 0.0114
    return lon-diff, lon+diff, lat-diff, lat+diff

# Determine the distribution of store types within a 1 km radius
def get_distribution(row, data, radius=1.0):
    # Apply a mask to find stores within the specified radius
    lon1, lon2, lat1, lat2 = get_border(row['Longitude'], row['Latitude'])
    condition = (data['좌표정보(x)'] > lon1) & (data['좌표정보(x)'] < lon2) & (data['좌표정보(y)'] > lat1) & (data['좌표정보(y)'] < lat2) # 예시 조건: 'column_name'이 3보다 큰 경우
    data = data[condition]

    mask = data.apply(lambda x: haversine(row['Longitude'], row['Latitude'], x['좌표정보(x)'], x['좌표정보(y)']) <= radius, axis=1)
    
    return mask

# pipeline file path
pipeline_filename = 'dataset/process_pipeline_jongro_1124.csv'

# Load the original dataset
original_data_path = 'dataset/updated_diningcode_jongro_1124.csv'
df = pd.read_csv(original_data_path)

# Filter the DataFrame to include only valid entries
filtered_df = df[(df['score'] > 0) & (df['category'].notnull())]

# Save the filtered DataFrame to a new CSV file (optional)
# filtered_data_path = '/Users/hwangyun/PycharmProjects/pred_loc/dataset/filtered_diningcode_output.csv'
# filtered_df.to_csv(filtered_data_path, index=False)

# Load the filtered and merged dataset
merged_data_path = "dataset/final_merged_filtered_jongro_data.csv"
data = pd.read_csv(merged_data_path)

# drop '통신판매업'
# mask = data['개방서비스명'] == '통신판매업'
# data = data[~mask]

# Fill missing '업태구분명' values with '개방서비스명'
data['업태구분명'] = data['업태구분명'].fillna(data['개방서비스명'])

# Use only the relevant columns for further processing
relevant_columns = ['좌표정보(x)', '좌표정보(y)', '업태구분명']
data = data[relevant_columns]

# Calculate the distribution of nearby stores for each row
# Note: This may be slow for large datasets; consider parallel processing methods if necessary
print(filtered_df)
# results = [get_distribution(row, data) for _, row in tqdm(filtered_df.iterrows(), total=filtered_df.shape[0])]
results = [get_distribution(row, data) for _, row in tqdm(filtered_df.iterrows(), total=filtered_df.shape[0])]
selected_indices = []
for mask in results:
    selected_indices.append([i for i, value in enumerate(mask) if value])


file = open(pipeline_filename, 'w', newline='', encoding='utf-8')

# CSV 파일을 쓰기 위한 writer 객체 생성
csv_writer = csv.writer(file)

for idx_list in selected_indices:
    string = ""
    for idx in idx_list:
        string += "{}".format(idx)
        if idx == idx_list[-1]:
            break
        string += "/"
    csv_writer.writerow([string])
