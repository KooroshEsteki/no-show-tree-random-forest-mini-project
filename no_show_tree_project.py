import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# 
# STEP 1 + STEP 2: Read the dataset into a Pandas DataFrame
# 

DATA_PATH = "KaggleV2-May-2016.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset shape:")
print(df.shape)

print("\nFirst five rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values per column:")
print(df.isna().sum())


# Drop missing values if any exist
df = df.dropna()

print("\nShape after dropping missing values:")
print(df.shape)


# 
# STEP 3: Feature Extraction
# Required features:
# Gender, Age, Scholarship, Hipertension, Diabetes,
# Alcoholism, Handcap, SMS_received
#
# Note:
# Hipertension = Hypertension
# Handcap = Handicap
# The spelling is kept because this is how it appears in the dataset.
# 

features = [
    "Gender",
    "Age",
    "Scholarship",
    "Hipertension",
    "Diabetes",
    "Alcoholism",
    "Handcap",
    "SMS_received"
]

target = "No-show"

# Keep only selected features and target
df = df[features + [target]]

# Remove invalid age values
# The dataset contains one row with Age = -1, which is not realistic.
df = df[df["Age"] >= 0]

print("\nShape after removing invalid age values:")
print(df.shape)

print("\nTarget distribution:")
print(df[target].value_counts())


# Convert target labels:
# No  = patient showed up
# Yes = patient did not show up
df[target] = df[target].map({
    "No": 0,
    "Yes": 1
})

print("\nTarget after encoding:")
print(df[target].value_counts())


X = df[features]
y = df[target]

print("\nSelected features:")
print(features)

print("\nFeature matrix shape:")
print(X.shape)

print("\nTarget shape:")
print(y.shape)


# 
# STEP 4: Preprocessing
# Scaling numeric features
# Encoding nominal features
# Dealing with NaN values
# 

numeric_features = [
    "Age",
    "Scholarship",
    "Hipertension",
    "Diabetes",
    "Alcoholism",
    "Handcap",
    "SMS_received"
]

categorical_features = [
    "Gender"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)


#
# STEP 5: Split the data
# 80% training set
# 10% validation set
# 10% test set
# 

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nTraining set:")
print(X_train.shape, y_train.shape)

print("\nValidation set:")
print(X_val.shape, y_val.shape)

print("\nTest set:")
print(X_test.shape, y_test.shape)


# 
# STEP 6: Training Tree-Based Classifiers
# Decision Tree Classifier
#
# Choose the best criterion by trying different values and
# validating performance on the validation set.
# 

criteria = ["gini", "entropy", "log_loss"]

decision_tree_results = []

for criterion in criteria:
    dt_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", DecisionTreeClassifier(
                criterion=criterion,
                random_state=42
            ))
        ]
    )

    dt_model.fit(X_train, y_train)

    y_val_pred = dt_model.predict(X_val)

    val_accuracy = accuracy_score(y_val, y_val_pred)

    decision_tree_results.append({
        "criterion": criterion,
        "validation_accuracy": val_accuracy
    })

    print(f"Criterion = {criterion:8s} | Validation accuracy = {val_accuracy:.4f}")


# Select best criterion
results_df = pd.DataFrame(decision_tree_results)
best_row = results_df.loc[results_df["validation_accuracy"].idxmax()]
best_criterion = best_row["criterion"]

print("\nDecision Tree validation results:")
print(results_df)

print("\nBest criterion:")
print(best_criterion)


# Train final decision tree model using the best criterion
final_dt_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(
            criterion=best_criterion,
            random_state=42
        ))
    ]
)

final_dt_model.fit(X_train, y_train)

y_test_pred_dt = final_dt_model.predict(X_test)

dt_test_accuracy = accuracy_score(y_test, y_test_pred_dt)

print("\nFinal Decision Tree Test Accuracy:")
print(round(dt_test_accuracy, 4))

print("\nDecision Tree Confusion Matrix:")
print("Labels: 0 = Showed up, 1 = No-show")
print(confusion_matrix(y_test, y_test_pred_dt, labels=[0, 1]))

print("\nDecision Tree Classification Report:")
print(classification_report(
    y_test,
    y_test_pred_dt,
    target_names=["Showed up", "No-show"]
))


# 
# STEP 7: Random Forest
#
# Repeat step 6 using Random Forest.
# Increase/decrease the number of estimators and compare
# classification metrics.
# 

n_estimators_values = [10, 50, 100, 200]

rf_results = []

for n in n_estimators_values:
    rf_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=n,
                random_state=42
            ))
        ]
    )

    rf_model.fit(X_train, y_train)

    y_val_pred_rf = rf_model.predict(X_val)
    y_test_pred_rf = rf_model.predict(X_test)

    val_accuracy = accuracy_score(y_val, y_val_pred_rf)
    test_accuracy = accuracy_score(y_test, y_test_pred_rf)

    rf_results.append({
        "n_estimators": n,
        "validation_accuracy": val_accuracy,
        "test_accuracy": test_accuracy
    })

    print("\n====================================================")
    print(f"Random Forest with n_estimators = {n}")
    print("====================================================")

    print("Validation Accuracy:")
    print(round(val_accuracy, 4))

    print("Test Accuracy:")
    print(round(test_accuracy, 4))

    print("\nConfusion Matrix:")
    print("Labels: 0 = Showed up, 1 = No-show")
    print(confusion_matrix(y_test, y_test_pred_rf, labels=[0, 1]))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_test_pred_rf,
        target_names=["Showed up", "No-show"]
    ))


rf_results_df = pd.DataFrame(rf_results)

print("\nRandom Forest comparison:")
print(rf_results_df)


# Choose the best random forest based on validation accuracy
best_rf_row = rf_results_df.loc[rf_results_df["validation_accuracy"].idxmax()]
best_n_estimators = int(best_rf_row["n_estimators"])

print("\nBest number of estimators for Random Forest:")
print(best_n_estimators)

print("\nComment:")
print(
    "Increasing the number of estimators usually makes the Random Forest model "
    "more stable. In this dataset, the accuracy changes only slightly when the "
    "number of estimators increases, so the improvement is not very large."
)
