import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, 
    roc_curve, auc, precision_recall_curve, f1_score,
    precision_score, recall_score
)
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

df = pd.read_csv('dataset/heart.csv')
X = df.drop('target', axis=1)
y = df['target']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

plt.figure(figsize=(10, 6))

train_count = len(y_train)
test_count = len(y_test)
total_count = train_count + test_count

train_percent = (train_count / total_count) * 100
test_percent = (test_count / total_count) * 100

train_sain = sum(y_train == 0)
train_malade = sum(y_train == 1)

test_sain = sum(y_test == 0)
test_malade = sum(y_test == 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

labels_split = ['Train', 'Test']
sizes = [train_count, test_count]
colors_split = ['#66b3ff', '#ff9999']
explode = (0.05, 0.05)

axes[0].pie(sizes, explode=explode, labels=labels_split, colors=colors_split,
           autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*total_count)})',
           shadow=True, startangle=90)
axes[0].set_title('Répartition Train/Test (80%/20%)', fontsize=14, fontweight='bold')

categories = ['Sain (Train)', 'Malade (Train)', 'Sain (Test)', 'Malade (Test)']
values = [train_sain, train_malade, test_sain, test_malade]
colors_bars = ['lightblue', 'lightcoral', 'blue', 'red']

bars = axes[1].bar(categories, values, color=colors_bars, alpha=0.8)
axes[1].set_title('Distribution des Classes par Ensemble', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Nombre de patients')
axes[1].set_ylim(0, max(values) * 1.1)

for bar, value in zip(bars, values):
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{value}', ha='center', va='bottom', fontweight='bold')

test_total = test_sain + test_malade
if test_total > 0:
    axes[1].text(2.5, test_sain/2, f'{test_sain/test_total*100:.1f}%', 
                ha='center', va='center', fontweight='bold', color='white')
    axes[1].text(3.5, test_malade/2, f'{test_malade/test_total*100:.1f}%', 
                ha='center', va='center', fontweight='bold', color='white')

plt.suptitle('Split des Données - Stratification Préservée', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('data_split_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

print("="*50)
print("STATISTIQUES DU SPLIT DES DONNÉES")
print("="*50)
print(f"Total patients: {total_count}")
print(f"\nEnsemble d'entraînement: {train_count} patients ({train_percent:.1f}%)")
print(f"  - Sain: {train_sain} ({train_sain/train_count*100:.1f}%)")
print(f"  - Malade: {train_malade} ({train_malade/train_count*100:.1f}%)")
print(f"\nEnsemble de test: {test_count} patients ({test_percent:.1f}%)")
print(f"  - Sain: {test_sain} ({test_sain/test_count*100:.1f}%)")
print(f"  - Malade: {test_malade} ({test_malade/test_count*100:.1f}%)")


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm_poly = SVC(kernel='poly', degree=3, random_state=42, probability=True)
svm_poly.fit(X_train_scaled, y_train)


y_pred = svm_poly.predict(X_test_scaled)
y_pred_proba = svm_poly.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
plt.title('Matrice de Confusion - SVM Polynomial')
plt.xlabel('Prédictions')
plt.ylabel('Vérité Terrain')
plt.tight_layout()
plt.savefig('confusion_matrix_svm_poly.png', dpi=300)
plt.show()

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Taux de Faux Positifs')
plt.ylabel('Taux de Vrais Positifs')
plt.title('Courbe ROC - SVM Polynomial')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve_svm_poly.png', dpi=300)
plt.show()

precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(recall_curve, precision_curve, color='green', lw=2)
plt.xlabel('Recall (Sensibilité)')
plt.ylabel('Precision')
plt.title('Courbe Precision-Recall - SVM Polynomial')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('precision_recall_svm_poly.png', dpi=300)
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [accuracy, precision, recall, f1]
colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold']

bars = ax.bar(metrics, values, color=colors, alpha=0.8)
ax.set_ylim([0, 1.1])
ax.set_title('Métriques de Performance - SVM Polynomial')
ax.set_ylabel('Score')

for bar, value in zip(bars, values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
           f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('performance_metrics_svm_poly.png', dpi=300)
plt.show()


cv_scores = cross_val_score(svm_poly, X_train_scaled, y_train, 
                           cv=5, scoring='accuracy')

plt.figure(figsize=(8, 6))
plt.bar(range(1, 6), cv_scores, color='lightblue', alpha=0.8)
plt.axhline(y=cv_scores.mean(), color='red', linestyle='--', 
            label=f'Moyenne = {cv_scores.mean():.3f}')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('Validation Croisée 5-Fold - SVM Polynomial')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim([0.7, 1.0])
plt.tight_layout()
plt.savefig('cross_validation_svm_poly.png', dpi=300)
plt.show()