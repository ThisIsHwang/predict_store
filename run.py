from utils import load_data
from model import ModelForPredictStoreType, ModelForPredictStoreHashes


def main():
    data_for_type, data_for_hash = load_data()
    #
    # model_store_type = ModelForPredictStoreType()
    # X, y = model_store_type.preprocess_data(data_for_type)
    # mean_scores = model_store_type.train_and_evaluate(X, y)
    #
    # print(f"Accuracy: {mean_scores[0]}")
    # print(f"F1 Score (Micro): {mean_scores[1]}")
    # print(f"F1 Score (Macro): {mean_scores[2]}")
    # print(f"F1 Score (Weighted): {mean_scores[3]}")

    model_store_hashes = ModelForPredictStoreHashes()
    X_hashes, y_hashes = model_store_hashes.preprocess_data(data_for_hash)
    mean_scores_hashes = model_store_hashes.train_and_evaluate(X_hashes, y_hashes)

    print(f"Hashes Classification - Jaccard Similarity Score: {mean_scores_hashes}")


if __name__ == '__main__':
    main()
