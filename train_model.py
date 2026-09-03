import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv("data/ai4i2020.csv")

# Remove unnecessary identification columns
df = df.drop(["UDI", "Product ID"], axis=1)

# Convert machine type into numbers
df = pd.get_dummies(df, columns=["Type"], drop_first=True)

# Separate input features and target
X = df.drop(
    [
        "Machine failure",
        "TWF",
        "HDF",
        "PWF",
        "OSF",
        "RNF"
    ],
    axis=1
)
y = df["Machine failure"]

# Split data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create the Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Check accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
# Save the trained model
joblib.dump(model, "models/predictive_maintenance_model.pkl")

print("\nModel saved successfully!")