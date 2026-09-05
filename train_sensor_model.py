import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


# ==================================================
# LOAD DATASET
# ==================================================

print("\nLoading sensor dataset...")

df = pd.read_csv("data/sensor.csv")


# ==================================================
# DATASET INFORMATION
# ==================================================

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())


# ==================================================
# TARGET COLUMN
# ==================================================

TARGET_COLUMN = "machine_status"


# ==================================================
# CHECK MACHINE STATUS VALUES
# ==================================================

print("\n" + "=" * 60)
print("MACHINE STATUS DISTRIBUTION")
print("=" * 60)

print(df[TARGET_COLUMN].value_counts())


# ==================================================
# REMOVE TARGET FROM FEATURES
# ==================================================

X = df.drop(columns=[TARGET_COLUMN])

y = df[TARGET_COLUMN]


# ==================================================
# KEEP ONLY NUMERIC SENSOR COLUMNS
# ==================================================

X = X.select_dtypes(include=["number"])


print("\n" + "=" * 60)
print("FEATURE INFORMATION")
print("=" * 60)

print("\nNumber of Features:")
print(X.shape[1])

print("\nFeature Names:")
print(X.columns.tolist())


# ==================================================
# HANDLE MISSING TARGET VALUES
# ==================================================

valid_rows = y.notna()

X = X.loc[valid_rows]
y = y.loc[valid_rows]


# ==================================================
# ENCODE MACHINE STATUS
# ==================================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)


print("\n" + "=" * 60)
print("MACHINE STATUS CLASSES")
print("=" * 60)

for index, class_name in enumerate(label_encoder.classes_):

    print(f"{index} -> {class_name}")


# ==================================================
# TRAIN / TEST SPLIT
# ==================================================

print("\nSplitting dataset into training and testing data...")


X_train, X_test, y_train, y_test = train_test_split(

    X,

    y_encoded,

    test_size=0.20,

    random_state=42,

    stratify=y_encoded

)


print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)


# ==================================================
# MACHINE LEARNING PIPELINE
# ==================================================

print("\nCreating machine learning model...")


model = Pipeline(

    [

        (

            "imputer",

            SimpleImputer(

                strategy="median"

            )

        ),

        (

            "classifier",

            RandomForestClassifier(

                n_estimators=150,

                max_depth=20,

                min_samples_split=5,

                min_samples_leaf=2,

                random_state=42,

                n_jobs=-1,

                class_weight="balanced"

            )

        )

    ]

)


# ==================================================
# TRAIN MODEL
# ==================================================

print("\nTraining model...")
print("This may take a few minutes because your dataset has over 220,000 rows.\n")


model.fit(

    X_train,

    y_train

)


# ==================================================
# MAKE PREDICTIONS
# ==================================================

print("Testing trained model...\n")


predictions = model.predict(

    X_test

)


# ==================================================
# MODEL ACCURACY
# ==================================================

accuracy = accuracy_score(

    y_test,

    predictions

)


print("\n" + "=" * 60)
print("MODEL RESULTS")
print("=" * 60)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")


# ==================================================
# CLASSIFICATION REPORT
# ==================================================

print("\nClassification Report:\n")


print(

    classification_report(

        y_test,

        predictions,

        target_names=label_encoder.classes_,

        zero_division=0

    )

)


# ==================================================
# SAVE TRAINED MODEL
# ==================================================

print("\nSaving trained model...")


joblib.dump(

    model,

    "models/sensor_machine_model.pkl"

)


# ==================================================
# SAVE FEATURE NAMES
# ==================================================

joblib.dump(

    X.columns.tolist(),

    "models/sensor_machine_features.pkl"

)


# ==================================================
# SAVE LABEL ENCODER
# ==================================================

joblib.dump(

    label_encoder,

    "models/sensor_machine_label_encoder.pkl"

)


# ==================================================
# FINISHED
# ==================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nFiles Created:")

print(
    "\nmodels/sensor_machine_model.pkl"
)

print(
    "models/sensor_machine_features.pkl"
)

print(
    "models/sensor_machine_label_encoder.pkl"
)

print("\nYour Sensor Machine AI model is ready!")