# Machine Learning (ML) Algorithm Development in Python
## Complete Step-by-Step Tutorial (Supervised Learning)

This tutorial guides you through the process of building a Machine Learning project from scratch. You will build, train, and evaluate a **K-Nearest Neighbors (KNN) Classifier** using Python's industry-standard **scikit-learn** library.

---

### Step 1: Set Up the Environment
To practice machine learning in Python, you need to install the core data science and ML libraries. Run the following command in your terminal:

```bash
pip install numpy pandas scikit-learn
```

---

### Step 2: Understand the Goal & Dataset
We will create a model that predicts whether a student will **Pass (1)** or **Fail (0)** an exam based on two features:
1. **Hours Studied**
2. **Attendance Percentage (%)**

---

### Step 3: Write the Python Code

Create a file named `ml_tutorial.py` and add the following complete, production-ready code:

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------------------------------------
# PHASE 1: Collect & Prepare Mock Data
# -------------------------------------------------------------
print("--- Phase 1: Preparing Data ---")
data = {
    'Hours_Studied': [4.5, 1.2, 7.8, 3.4, 9.0, 2.1, 6.5, 5.0, 8.2, 1.5],
    'Attendance':    [85,  40,  95,  60,  98,  55,  88,  75,  92,  45],
    'Result':        [1,   0,   1,   0,   1,   0,   1,   1,   1,   0]  # 1 = Pass, 0 = Fail
}

df = pd.DataFrame(data)
print("Dataset Sample:\n", df.head())

# Split into Features (X) and Target Label (y)
X = df[['Hours_Studied', 'Attendance']]
y = df['Result']

# -------------------------------------------------------------
# PHASE 2: Split Data for Training & Testing
# -------------------------------------------------------------
# We use 80% of data for training and 20% for testing the accuracy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------------------------------------
# PHASE 3: Feature Scaling (Normalization)
# -------------------------------------------------------------
# Distance-based models like KNN perform poorly if scales are drastically different 
# (e.g., comparing Hours [0-10] to Attendance [0-100])
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------------------------------------
# PHASE 4: Initialize and Train the Model
# -------------------------------------------------------------
print("\n--- Phase 4: Training KNN Classifier ---")
# n_neighbors=3 means the model looks at the 3 closest data points to make a decision
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train_scaled, y_train)
print("Model trained successfully.")

# -------------------------------------------------------------
# PHASE 5: Evaluate the Algorithm
# -------------------------------------------------------------
print("\n--- Phase 5: Evaluating Model Performance ---")
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {accuracy * 100:.2f}%")
print("\nDetailed Classification Report:\n", classification_report(y_test, y_pred))

# -------------------------------------------------------------
# PHASE 6: Make Predictions on New/Unseen Data
# -------------------------------------------------------------
print("\n--- Phase 6: Making Live Predictions ---")
# Example Student: Studied 5.5 hours, 80% Attendance
new_student = np.array([[5.5, 80]])
new_student_scaled = scaler.transform(new_student)

prediction = model.predict(new_student_scaled)
probability = model.predict_proba(new_student_scaled)

if prediction[0] == 1:
    print(f"Prediction: Student will PASS! (Confidence: {probability[0][1]*100:.1f}%)")
else:
    print(f"Prediction: Student will FAIL. (Confidence: {probability[0][0]*100:.1f}%)")
```

---

### Step 4: The 6 Pillars of the ML Pipeline Explained

| Stage | What Happens | Why It Matters |
| :--- | :--- | :--- |
| **1. Data Preparation** | Raw features and labels are converted into numeric formats (like arrays or DataFrames). | Algorithms only process structured numbers, not raw real-world concepts. |
| **2. Train/Test Split** | Data is split into isolated chunks (e.g., 80% train, 20% test). | Validates whether the model actually learned patterns or just memorized data. |
| **3. Feature Scaling** | Values are shifted so they share a common scale (e.g., between -3 and +3). | Prevents large numbers (like 95% attendance) from drowning out small numbers (like 4 hours). |
| **4. Model Fitting** | The `.fit()` function trains the algorithm mathematically on the training features. | This is the actual "learning" step where equations map inputs to outputs. |
| **5. Performance Metrics** | The test features are predicted and compared against real answers. | Gives exact insight into accuracy, precision, and error rates before deployment. |
| **6. Deployment Inference** | Production pipelines feed raw, live user entries to get real-time answers. | Generates the actual commercial value or automated decisions for the application. |

---

*This document was generated for educational purposes.*
