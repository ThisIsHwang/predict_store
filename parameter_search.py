from utils import load_data
from model import ModelForPredictStoreType, ModelForPredictStoreHashes
from sklearn.model_selection import GridSearchCV


def main():
    data_for_type, data_for_hash = load_data()

    # Model for Store Type
    model_store_type = ModelForPredictStoreType()
    X, y = model_store_type.preprocess_data(data_for_type)

    best_params_type, best_score_type = model_store_type.perform_grid_search(X, y)
    print("Best Parameters for Store Type:", best_params_type)
    print("Best Score for Store Type:", best_score_type)

    # Model for Store Hashes
    model_store_hashes = ModelForPredictStoreHashes()
    X_hashes, y_hashes = model_store_hashes.preprocess_data(data_for_hash)

    best_params_hashes, best_score_hashes = model_store_hashes.perform_grid_search(X_hashes, y_hashes)
    print("Best Parameters for Store Hashes:", best_params_hashes)
    print("Best Score for Store Hashes:", best_score_hashes)


if __name__ == '__main__':
    main()
