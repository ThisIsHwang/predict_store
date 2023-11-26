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

# Determine the distribution of store types within a 0.5 km radius
def get_distribution(row, data, radius=0.5):
    # Apply a mask to find stores within the specified radius
    mask = data.apply(lambda x: haversine(row['Longitude'], row['Latitude'], x['좌표정보(x)'], x['좌표정보(y)']) <= radius, axis=1)
    nearby_stores = data[mask]
    distribution = nearby_stores['업태구분명'].value_counts(normalize=True)
    return distribution.to_dict()

# Load the original dataset
original_data_path = '/Users/hwangyun/PycharmProjects/pred_loc/dataset/updated_diningcode_output.csv'
df = pd.read_csv(original_data_path)

# Filter the DataFrame to include only valid entries
filtered_df = df[(df['score'] > 0) & (df['category'].notnull())]

# Save the filtered DataFrame to a new CSV file (optional)
# filtered_data_path = '/Users/hwangyun/PycharmProjects/pred_loc/dataset/filtered_diningcode_output.csv'
# filtered_df.to_csv(filtered_data_path, index=False)

# Load the filtered and merged dataset
merged_data_path = "/Users/hwangyun/PycharmProjects/pred_loc/final_merged_filtered_data.csv"
data = pd.read_csv(merged_data_path)

# Fill missing '업태구분명' values with '개방서비스명'
data['업태구분명'] = data['업태구분명'].fillna(data['개방서비스명'])

# Use only the relevant columns for further processing
relevant_columns = ['좌표정보(x)', '좌표정보(y)', '업태구분명']
data = data[relevant_columns]

# Calculate the distribution of nearby stores for each row
# Note: This may be slow for large datasets; consider parallel processing methods if necessary
results = [get_distribution(row, data) for _, row in tqdm(filtered_df.iterrows(), total=filtered_df.shape[0])]

# Create a DataFrame from the distribution results and combine it with the filtered data
distribution_df = pd.DataFrame(results).fillna(0)
filtered_df = pd.concat([filtered_df.reset_index(drop=True), distribution_df], axis=1)

# Save the combined data with distribution information
intermediate_data_path = '/Users/hwangyun/PycharmProjects/pred_loc/dataset/inter_diningcode.csv'
filtered_df.to_csv(intermediate_data_path, index=False)

# Load the data for the machine learning model
data = pd.read_csv(intermediate_data_path)

# # Simplify the category labels by splitting on '>' and keeping the last element
# data['category'] = data['category'].str.split('>').str[-1]

# # Prepare the features (X) and labels (y) for the model
# feature_columns = ['category', 'Title', 'Latitude', 'Longitude', 'Info Title', 'Title Place', 'hashes', 'score', 'userScore', 'heart', 'title', 'address', 'roadAddress', 'mapx', 'mapy', 'categories']
# X = data.drop(columns=feature_columns)
# y = data['category']

# # Split the data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)