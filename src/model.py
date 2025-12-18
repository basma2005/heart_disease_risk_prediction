import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

df = pd.read_csv('dataset/heart.csv')
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)
accuracy_lr = accuracy_score(y_test, y_pred_lr)

print("SVM WITH DIFFERENT KERNELS:")

svm_linear = SVC(kernel='linear', random_state=42)
svm_linear.fit(X_train_scaled, y_train)
y_pred_linear = svm_linear.predict(X_test_scaled)
accuracy_linear = accuracy_score(y_test, y_pred_linear)
print(f"SVM Linéaire - Accuracy: {accuracy_linear:.4f}")

svm_poly = SVC(kernel='poly', degree=3, random_state=42)
svm_poly.fit(X_train_scaled, y_train)
y_pred_poly = svm_poly.predict(X_test_scaled)
accuracy_poly = accuracy_score(y_test, y_pred_poly)
print(f"SVM Polynomial (deg 3) - Accuracy: {accuracy_poly:.4f}")

svm_rbf = SVC(kernel='rbf', random_state=42)
svm_rbf.fit(X_train_scaled, y_train)
y_pred_rbf = svm_rbf.predict(X_test_scaled)
accuracy_rbf = accuracy_score(y_test, y_pred_rbf)
print(f"SVM RBF - Accuracy: {accuracy_rbf:.4f}")

rf_classifier = RandomForestClassifier(
    n_estimators=100,      
    max_depth=5,           
    min_samples_split=2,   
    min_samples_leaf=1,   
    max_features='sqrt',   
    random_state=42,
    n_jobs=-1              
)

rf_classifier.fit(X_train, y_train)

predictions = rf_classifier.predict(X_test)

accuracy_rf = accuracy_score(y_test, predictions)

print(f"\nCOMPARISON:")
print(f"Logistic Regression: {accuracy_lr:.4f}")
print(f"SVM Linear: {accuracy_linear:.4f}")
print(f"SVM Polynomial: {accuracy_poly:.4f}")
print(f"SVM RBF: {accuracy_rbf:.4f}")
print(f"Random Forest: {accuracy_rf:.4f}")

models = {
    'Logistic Regression': (lr_model, accuracy_lr),
    'SVM Linear': (svm_linear, accuracy_linear),
    'SVM Polynomial': (svm_poly, accuracy_poly),
    'SVM RBF': (svm_rbf, accuracy_rbf),
    'Random Forest': (rf_classifier,accuracy_rf)
}

best_model_name = max(models, key=lambda x: models[x][1])
best_model, best_accuracy = models[best_model_name]

print(f"\nBest Model: {best_model_name} - Accuracy: {best_accuracy:.4f}")

joblib.dump(best_model, 'model/best_heart_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')
print("Best model and scaler saved!")