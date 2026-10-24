import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# generate synthetic dataset
np.random.seed(42)
hours = np.random.uniform(0.5, 10.0, 100)
attendance = np.clip(hours * 8 + np.random.normal(25, 10, 100), 30, 100)

score = (hours * 10) + attendance
result = (score > 100).astype(int)

df = pd.DataFrame({'Hours': hours, 'Attendance': attendance, 'Score': score, 'Result': result})

print("Generated dataset sample (first 5 rows):")
print(df.head())

x = df[['Hours', 'Attendance']]
y = df['Result']

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.values)
X_test_scaled = scaler.transform(X_test.values)

models = {
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
}

x_min, x_max = x['Hours'].min() - 1, x['Hours'].max() + 1
y_min, y_max = x['Attendance'].min() - 1, x['Attendance'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
grid_pts = np.c_[xx.ravel(), yy.ravel()]

fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
results = []

for ax, (name, model) in zip(axes, models.items()):
    if name == 'Decision Tree':
        model.fit(X_train.values, y_train)
        train_pred = model.predict(X_train.values)
        test_pred = model.predict(X_test.values)
        Z = model.predict(grid_pts)
    else:
        model.fit(X_train_scaled, y_train)
        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)
        Z = model.predict(scaler.transform(grid_pts))

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    results.append({'Model': name, 'Train Accuracy': train_acc, 'Test Accuracy': test_acc})

    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    ax.scatter(x['Hours'], x['Attendance'], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    ax.set_title(f"{name}\nTrain accuracy: {train_acc:.1%} | Test accuracy: {test_acc:.1%}")
    ax.set_xlabel('Hours Studied')
    ax.set_ylabel('Attendance (%)')

results_df = pd.DataFrame(results).sort_values('Test Accuracy', ascending=False)
print("\nModel performance:")
print(results_df.to_string(index=False, float_format='{:.2%}'.format))
best = results_df.iloc[0]
print(f"\nBest model: {best['Model']} (test accuracy {best['Test Accuracy']:.1%})")

output_path = 'model_comparison.png'
fig.savefig(output_path, dpi=150)
print(f"Saved decision boundary plot to {output_path}")

plt.show()
