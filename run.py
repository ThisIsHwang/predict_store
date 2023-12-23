import pandas as pd
from model import Model

def load_data():
    data_folder = "data/"
    youngdeungpo_data = pd.read_csv(data_folder + 'inter_diningcode_youngdeungpo_dropped.csv')
    jongro_data = pd.read_csv(data_folder + 'inter_diningcode_jongro_dropped.csv')
    youngdeungpo_hashes = pd.read_csv(data_folder + 'inter_hashes_youngdeungpo_dropped.csv')
    jongro_hashes = pd.read_csv(data_folder + 'inter_hashes_jongro_dropped.csv')

    # Data preprocessing
    columns_to_drop = ['categories', 'Info Title', 'Title Place', 'hashes', 'score', 'userScore', 'heart', 'title',
                       'address', 'roadAddress', 'mapx', 'mapy']
    youngdeungpo_data.drop(columns=columns_to_drop, inplace=True)
    jongro_data.drop(columns=columns_to_drop, inplace=True)
    youngdeungpo_hashes.drop(columns=columns_to_drop, inplace=True)
    jongro_hashes.drop(columns=columns_to_drop, inplace=True)

    # Merge and remove duplicates
    youngdeungpo_data = pd.merge(youngdeungpo_data, youngdeungpo_hashes,
                                 on=['Title', 'Latitude', 'Longitude', 'category'], how='inner')
    jongro_data = pd.merge(jongro_data, jongro_hashes, on=['Title', 'Latitude', 'Longitude', 'category'], how='inner')

    youngdeungpo_data.drop_duplicates(subset=['Title', 'Latitude', 'Longitude', 'category'], inplace=True)
    jongro_data.drop_duplicates(subset=['Title', 'Latitude', 'Longitude', 'category'], inplace=True)

    # Concatenate data
    data = pd.concat([youngdeungpo_data, jongro_data], ignore_index=True)

    # Final preprocessing steps (if any)
    data['category'] = data['category'].str.split('>').str[-1]

    return data

def main():
    data = load_data()
    model = Model()
    X, y = model.preprocess_data(data)
    mean_scores = model.train_and_evaluate(X, y)

    print(f"Accuracy: {mean_scores[0]}")
    print(f"F1 Score (Micro): {mean_scores[1]}")
    print(f"F1 Score (Macro): {mean_scores[2]}")
    print(f"F1 Score (Weighted): {mean_scores[3]}")

if __name__ == '__main__':
    main()
