import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset - IMPORTANT: No header row!
df = pd.read_csv("heart.csv", header=None)

# Rename columns properly
df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
              'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

# Replace '?' with 0 (clean missing data)
df = df.replace('?', 0).infer_objects(copy=False)

# Convert target to binary (0 = no disease, 1 = disease)
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(df.head())

# Separate features and target
X = df.drop("target", axis=1)
y = df["target"]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2%}")

# Test a single prediction
sample = X_test.iloc[[0]]
prediction = model.predict(sample)[0]
probability = model.predict_proba(sample)[0]
print(f"\nSample prediction: {prediction}")
print(f"Probability array: {probability}")
print("Notice: these raw numbers mean nothing without context!")