import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, cross_val_predict
# Information
# - Baseline: No PCA, SVM
# - Dataset: get_weighted_distribution() 함수 적용

# 데이터 로드 및 전처리
data = pd.read_csv('predict_store\dataset\inter_diningcode.csv')  # Update the path to your file
data['category'] = data['category'].str.split('>').str[-1]

# 특징 및 레이블 준비
X = data.drop(columns=[
    'category', 'Title', 'Latitude', 'Longitude', 'Info Title', 'Title Place',
    'hashes', 'score', 'userScore', 'heart', 'title', 'address', 'roadAddress',
    'mapx', 'mapy', 'categories'
])
y = data['category']

# 훈련 및 테스트 데이터 분할
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 데이터 표준화
scaler = StandardScaler()
#X_train_scaled = scaler.fit_transform(X_train)
#X_test_scaled = scaler.transform(X_test)

# SVM 모델 훈련 및 평가
svm_model = SVC()
#svm_model.fit(X_train_scaled, y_train)
kf = KFold(n_splits=3, shuffle=True, random_state=42)  #Cross Validation for k = 3
# Perform k-fold cross-validation
#cv_results_score = cross_val_score(svm_model, X_scaled, y, cv=kf)#getting cross validation scores
# Initialize an array to store predicted values
all_y_pred = np.array([])
scores = np.array([[], [], [], []])
# Iterate over each fold
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    X_train_Scaled = scaler.fit_transform(X_train)
    X_test_Scaled = scaler.transform(X_test)
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Fit the classifier on the training data
    svm_model.fit(X_train_Scaled, y_train)

    # Get predicted values for the test set in this fold
    fold_y_pred = svm_model.predict(X_test_Scaled)
    # Append the predicted values to the array
    all_y_pred = np.concatenate((all_y_pred, fold_y_pred))
    # 성능 측정
    all_scores = np.array([[accuracy_score(y_test, fold_y_pred)], [f1_score(y_test, fold_y_pred, average='micro')], [f1_score(y_test, fold_y_pred, average='macro')], [f1_score(y_test, fold_y_pred, average='weighted')]])
    scores = np.append(scores, all_scores, axis = 1)
# 결과 출력
mean_scores = np.mean(scores, axis = 1)
print(f"Accuracy: {mean_scores[0]}")
print(f"F1 Score (Micro): {mean_scores[1]}")
print(f"F1 Score (Macro): {mean_scores[2]}")
print(f"F1 Score (Weighted): {mean_scores[3]}")