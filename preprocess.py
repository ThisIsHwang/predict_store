import pandas as pd
from tqdm import tqdm
import math

# Haversine function remains the same
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

# Modified function to calculate weighted distribution of store types based on 'hashes'
def get_weighted_distribution(row, data, max_radius=1.0):
    weight_within_05 = 2
    weight_within_1 = 1

    # Calculate weights based on distance
    mask = data.apply(lambda x: haversine(row['Longitude'], row['Latitude'], x['Longitude'], x['Latitude']), axis=1)
    data['weight'] = mask.apply(lambda distance: weight_within_05 if distance <= 0.5 else (weight_within_1 if distance <= max_radius else 0))

    # Filter and process 'hashes'
    nearby_stores = data[data['weight'] > 0]
    nearby_stores = nearby_stores.explode('hashes')  # Splitting the hashes into separate rows
    weighted_counts = nearby_stores.groupby('hashes')['weight'].sum()
    total_weight = weighted_counts.sum()
    weighted_distribution = (weighted_counts / total_weight).to_dict()

    return weighted_distribution

# Load datasets
original_data_path = '/Users/hwangyun/PycharmProjects/predict_store/dataset/updated_diningcode_youngdeungpo_1124.csv'
df = pd.read_csv(original_data_path)
# data = pd.read_csv(merged_data_path, encoding='utf-8')

# Prepare 'hashes' column by splitting and removing empty strings
df['hashes'] = df['hashes'].str.split(',').apply(lambda x: [item.strip() for item in x if item.strip() != ''])


# Apply the modified function
results = [get_weighted_distribution(row, df) for _, row in tqdm(df.iterrows(), total=df.shape[0])]

# Process results as before
distribution_df = pd.DataFrame(results).fillna(0)
df = pd.concat([df.reset_index(drop=True), distribution_df], axis=1)
df.to_csv('/Users/hwangyun/PycharmProjects/predict_store/dataset/inter_hashes_youngdeungpo_dropped.csv', index=False)
# Save or further process the DataFrame
# df.to_csv(intermediate_data_path, index=False)
original_jongro_data_path = '/Users/hwangyun/PycharmProjects/predict_store/dataset/updated_diningcode_jongro_1124.csv'
df = pd.read_csv(original_jongro_data_path)
# data = pd.read_csv(merged_data_path, encoding='utf-8')

# Prepare 'hashes' column by splitting and removing empty strings
df['hashes'] = df['hashes'].str.split(',').apply(lambda x: [item.strip() for item in x if item.strip() != ''])


# Apply the modified function
results = [get_weighted_distribution(row, df) for _, row in tqdm(df.iterrows(), total=df.shape[0])]

# Process results as before
distribution_df = pd.DataFrame(results).fillna(0)
df = pd.concat([df.reset_index(drop=True), distribution_df], axis=1)
df.to_csv('/Users/hwangyun/PycharmProjects/predict_store/dataset/inter_hashes_jongro_dropped.csv', index=False)
