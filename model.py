import pandas as pd
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

class Model:
    def __init__(self):
        self.scaler = StandardScaler()
        self.base_classifier = DecisionTreeClassifier()
        self.bagging_classifier = BaggingClassifier(self.base_classifier, n_estimators=10, random_state=42, n_jobs=-1)
        self.kf = KFold(n_splits=3, shuffle=True, random_state=42)

    def preprocess_data(self, data):
        data = data.dropna(axis=1)
        X = data.drop(columns=['category', 'Title', 'Latitude', 'Longitude'])
        y = data['category']
        return X, y

    def train_and_evaluate(self, X, y):
        all_y_pred = np.array([])
        scores = np.array([[], [], [], []])

        for train_index, test_index in self.kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            self.bagging_classifier.fit(X_train_scaled, y_train)
            fold_y_pred = self.bagging_classifier.predict(X_test_scaled)

            all_y_pred = np.concatenate((all_y_pred, fold_y_pred))
            all_scores = np.array([
                [accuracy_score(y_test, fold_y_pred)],
                [f1_score(y_test, fold_y_pred, average='micro')],
                [f1_score(y_test, fold_y_pred, average='macro')],
                [f1_score(y_test, fold_y_pred, average='weighted')]
            ])
            scores = np.append(scores, all_scores, axis=1)

        mean_scores = np.mean(scores, axis=1)
        return mean_scores