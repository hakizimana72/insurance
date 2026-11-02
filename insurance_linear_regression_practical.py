"""
INSURANCE CHARGES PREDICTION 

"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATA_PATH = Path("insurance.csv")  # Change this if your CSV is elsewhere

df = pd.read_csv(DATA_PATH)

print("\nQUESTION 6: FIRST FIVE ROWS")
print(df.head())

print("\nDATASET INFORMATION")
df.info()

print("\nSUMMARY STATISTICS")
print(df.describe())

print("\nMISSING VALUES")
print(df.isnull().sum())

# Check that the expected columns are present.
required_columns = ["age", "sex", "bmi", "children", "smoker", "region", "charges"]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Dataset is missing expected columns: {missing_columns}")

# Remove rows with missing values in the columns used in this practical.
df = df[required_columns].dropna().copy()

# ------------------------------------------------------------
# QUESTION 7: ONE-HOT ENCODING OF CATEGORICAL VARIABLES
# ------------------------------------------------------------

df_encoded = pd.get_dummies(
    df,
    columns=["sex", "smoker", "region"],
    drop_first=True,
    dtype=int
)

print("\nQUESTION 7: ENCODED DATASET")
print(df_encoded.head())
print("\nENCODED COLUMN NAMES")
print(df_encoded.columns.tolist())

# ------------------------------------------------------------
# QUESTION 8: SPLIT FEATURES (X) AND TARGET (y), 80:20
# ------------------------------------------------------------

X = df_encoded.drop(columns=["charges"])
y = df_encoded["charges"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print("\nQUESTION 8: DATA SPLIT")
print("Training features:", X_train.shape)
print("Testing features: ", X_test.shape)
print("Training target:  ", y_train.shape)
print("Testing target:   ", y_test.shape)

# ------------------------------------------------------------
# QUESTION 9: BUILD AND TRAIN LINEAR REGRESSION MODEL
# ------------------------------------------------------------

model = LinearRegression()
model.fit(X_train, y_train)

print("\nQUESTION 9: MODEL COEFFICIENTS")
for feature, coefficient in zip(X.columns, model.coef_):
    print(f"{feature}: {coefficient:.4f}")

print("\nModel intercept:", round(model.intercept_, 4))

# ------------------------------------------------------------
# QUESTION 10: DISPLAY THE LINEAR REGRESSION EQUATION
# ------------------------------------------------------------

equation = f"charges = {model.intercept_:.4f}"
for feature, coefficient in zip(X.columns, model.coef_):
    sign = "+" if coefficient >= 0 else "-"
    equation += f" {sign} {abs(coefficient):.4f}({feature})"

print("\nQUESTION 10: REGRESSION EQUATION")
print(equation)

# ------------------------------------------------------------
# QUESTION 11: EVALUATE THE MODEL
# ------------------------------------------------------------

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nQUESTION 11: MODEL EVALUATION")
print(f"Mean Absolute Error (MAE): {mae:,.2f}")
print(f"Mean Squared Error (MSE): {mse:,.2f}")
print(f"R-squared (R²): {r2:.4f}")
print(f"R² as a percentage: {r2 * 100:.2f}%")

# ------------------------------------------------------------
# QUESTION 12(i): ACTUAL-VS-PREDICTED REGRESSION PLOT
# ------------------------------------------------------------

plt.figure(figsize=(9, 6))
plt.scatter(y_test, y_pred, alpha=0.65, edgecolors="none", label="Test observations")

line_min = min(y_test.min(), y_pred.min())
line_max = max(y_test.max(), y_pred.max())
plt.plot(
    [line_min, line_max],
    [line_min, line_max],
    linestyle="--",
    linewidth=2,
    label="Perfect prediction"
)

plt.xlabel("Actual insurance charges ($)")
plt.ylabel("Predicted insurance charges ($)")
plt.title("Linear Regression: Actual vs Predicted Insurance Charges")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("regression_actual_vs_predicted.png", dpi=300)
plt.show()

# ------------------------------------------------------------
# QUESTION 12(ii): REGRESSION-APPROPRIATE RESIDUAL PLOT
# A confusion matrix is for classification, not regression.
# ------------------------------------------------------------

residuals = y_test - y_pred

plt.figure(figsize=(9, 6))
sns.scatterplot(x=y_pred, y=residuals, alpha=0.65)
plt.axhline(y=0, linestyle="--", linewidth=2)
plt.xlabel("Predicted insurance charges ($)")
plt.ylabel("Residual (actual - predicted) ($)")
plt.title("Residual Plot for Linear Regression")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("regression_residual_plot.png", dpi=300)
plt.show()

# ------------------------------------------------------------
# QUESTION 13: PREDICT CHARGES FOR THE GIVEN NEW INDIVIDUAL
# age=19, sex=female, bmi=27.9, children=0,
# smoker=yes, region=southwest
# ------------------------------------------------------------

new_person = pd.DataFrame([{
    "age": 19,
    "sex": "female",
    "bmi": 27.9,
    "children": 0,
    "smoker": "yes",
    "region": "southwest"
}])

# Encode the new record and align its columns exactly with model training.
new_person_encoded = pd.get_dummies(
    new_person,
    columns=["sex", "smoker", "region"],
    drop_first=True,
    dtype=int
)

new_person_encoded = new_person_encoded.reindex(
    columns=X.columns,
    fill_value=0
)

predicted_charge = model.predict(new_person_encoded)[0]

print("\nQUESTION 13: NEW INDIVIDUAL")
print(new_person.to_string(index=False))
print(f"\nPredicted insurance charge: ${predicted_charge:,.2f}")

# Save actual-vs-predicted test results for reference.
results = pd.DataFrame({
    "actual_charges": y_test.values,
    "predicted_charges": y_pred
})
results.to_csv("insurance_prediction_results.csv", index=False)

print("\nFinished. Graphs and prediction results have been saved in this folder.")
