import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, jaccard_score
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from concurrent.futures import ProcessPoolExecutor
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
# using bagging classifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

class ModelForPredictStoreType:
    def __init__(self):
        self.scaler = StandardScaler()
        self.num_classes = None  # This will hold the number of unique classes
        #self.xgb_classifier = xgb.XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, eval_metric='mlogloss')
        self.xgb_classifier = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=100, random_state=42, n_jobs=-1)
        self.kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    def preprocess_data(self, data):
        data = data.dropna(axis=1)
        X = data.drop(columns=['category', 'Title', 'Latitude', 'Longitude'])
        y = data['category']
        # Count the occurrences of each class
        class_counts = y.value_counts()

        # Identify classes with more than one instance
        valid_classes = class_counts[class_counts > 1].index

        # Filter the data to include only valid classes
        valid_data = data[data['category'].isin(valid_classes)]

        # Update X and y after filtering
        X = valid_data.drop(columns=['category', 'Title', 'Latitude', 'Longitude'])
        y = valid_data['category']

        # Encoding the labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)

        return X, y_encoded

    def _train_evaluate_fold(self, indices):
        train_index, test_index = indices
        X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
        y_train, y_test = self.y[train_index], self.y[test_index]

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.xgb_classifier.fit(X_train_scaled, y_train)
        fold_y_pred = self.xgb_classifier.predict(X_test_scaled)

        return y_test, fold_y_pred

    def train_and_evaluate(self, X, y):
        self.X = X
        self.y = y
        indices = list(self.kf.split(X,y))
        scores = []

        with ProcessPoolExecutor() as executor:
            for y_test, fold_y_pred in executor.map(self._train_evaluate_fold, indices):
                scores.append([
                    accuracy_score(y_test, fold_y_pred),
                    f1_score(y_test, fold_y_pred, average='micro'),
                    f1_score(y_test, fold_y_pred, average='macro'),
                    f1_score(y_test, fold_y_pred, average='weighted')
                ])

        mean_scores = np.mean(scores, axis=0)
        return mean_scores
class ModelForPredictStoreHashes:
    def __init__(self):
        self.scaler = StandardScaler()
        self.classifier = OneVsRestClassifier(LogisticRegression(max_iter=1000,random_state=42, n_jobs=-1), n_jobs=-1)
        self.kf = KFold(n_splits=3, shuffle=True, random_state=42)

    def preprocess_data(self, data):
        data = data.dropna(axis=1)
        data = pd.get_dummies(data, columns=['category'])
        label_columns = [col for col in data.columns if col.startswith('whether_')]

        y = data[label_columns]
        X = data.drop(columns=label_columns + ['Title', 'Latitude', 'Longitude'])
        return X, y

    def _train_evaluate_fold(self, indices):
        train_index, test_index = indices
        X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
        y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.classifier.fit(X_train_scaled, y_train)
        y_pred = self.classifier.predict(X_test_scaled)
        return jaccard_score(y_test, y_pred, average='samples')

    def train_and_evaluate(self, X, y):
        self.X = X
        self.y = y
        indices = list(self.kf.split(X, y))
        jaccard_scores = []

        with ProcessPoolExecutor() as executor:
            for jaccard_score_value in executor.map(self._train_evaluate_fold, indices):
                jaccard_scores.append(jaccard_score_value)

        mean_jaccard = np.mean(jaccard_scores)
        return mean_jaccard
