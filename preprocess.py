import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm



from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np
import math

# Function to calculate distance using the haversine formula
def haversine(lon1, lat1, lon2, lat2):
    R = 6371  # Radius of Earth in kilometers
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = (math.sin(dLat / 2) * math.sin(dLat / 2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dLon / 2) * math.sin(dLon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance


# Function to get distribution of nearby stores
def get_distribution(row, data):
    mask = data.apply(lambda x: haversine(row['Longitude'], row['Latitude'], x['좌표정보(x)'], x['좌표정보(y)']) <= 0.5, axis=1)
    nearby_stores = data[mask]
    distribution = nearby_stores['업태구분명'].value_counts(normalize=True)
    return distribution.to_dict()

# Load the CSV file into a DataFrame
df = pd.read_csv('dataset/diningcode_with_naver_label.csv')

# Filter the DataFrame
# Ensure 'score' is greater than 0 and 'category' is not null
filtered_df = df[(df['score'] > 0) & (df['category'].notnull())]

#Save the filtered DataFrame to a new CSV file
#filtered_df.to_csv('/Users/hwangyun/PycharmProjects/pred_loc/dataset/filtered_diningcode_output.csv', index=False)

# Read the CSV file
data = pd.read_csv("dataset/nearby_stores.csv")

# Fill missing '업태구분명' values with '개방서비스명'
data['업태구분명'] = data['업태구분명'].fillna(data['개방서비스명'])


# Assume '경도', '위도' are longitude and latitude columns respectively
columns = ['좌표정보(x)', '좌표정보(y)', '업태구분명']
data = data[columns]


# Here we'll just use a simple loop for demonstration
# Note: This can be very slow for large datasets. Use parallel processing as shown above.
results = []
for _, row in tqdm(filtered_df.iterrows()):
    results.append(get_distribution(row, data))


distribution_df = pd.DataFrame(results).fillna(0)
filtered_df = filtered_df.reset_index(drop=True)
filtered_df = pd.concat([filtered_df, distribution_df], axis=1)
filtered_df.to_csv('dataset/preprocessed_diningcode.csv', index=False)