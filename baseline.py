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
from sklearn.model_selection import cross_val_score
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 데이터 표준화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SVM 모델 훈련 및 평가
svm_model = SVC()
svm_model.fit(X_train_scaled, y_train)
kf = KFold(n_splits=3, shuffle=True, random_state=42)  # You can set the random_state for reproducibility
# Perform k-fold cross-validation
cv_results = cross_val_score(svm_model, X, y, cv=kf)
#pca_scores[i, j] = accuracy_score(y_test, pca_y_pred_test)

y_pred_test = svm_model.predict(X_test_scaled)

# 성능 측정
#accuracy = accuracy_score(y_test, y_pred_test)
accuracy = np.mean(cv_results)
f1_micro = f1_score(y_test, y_pred_test, average='micro')
f1_macro = f1_score(y_test, y_pred_test, average='macro')
f1_weighted = f1_score(y_test, y_pred_test, average='weighted')

# 결과 출력
print(f"Accuracy: {accuracy}")
print(f"F1 Score (Micro): {f1_micro}")
print(f"F1 Score (Macro): {f1_macro}")
print(f"F1 Score (Weighted): {f1_weighted}")