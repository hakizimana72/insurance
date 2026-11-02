INSURANCE MACHINE LEARNING — COMPLETE ANACONDA PACKAGE

CONTENTS
- Insurance_Charges_Prediction.ipynb: ready-to-open Jupyter Notebook
- insurance_linear_regression_practical.py: standalone Python script (if included)
- Insurance_Charges_Machine_Learning_Assignment.docx: written assignment (if included)
- requirements.txt: required Python packages

DATASET
The package does not include insurance.csv because you have not uploaded the dataset here.
Download the insurance.csv file provided by your lecturer and place it in this folder.

HOW TO USE WITH ANACONDA
1. Extract this ZIP file.
2. Put insurance.csv in the extracted folder beside the notebook.
3. Open Anaconda Navigator and launch Jupyter Notebook.
4. Browse to this extracted folder.
5. Open Insurance_Charges_Prediction.ipynb.
6. Run each cell from top to bottom with Shift + Enter.

INSTALL PACKAGES IF NEEDED
In Anaconda Prompt, run:
    pip install -r requirements.txt

The notebook produces model coefficients, intercept, regression equation, MAE, MSE,
R-squared, two graphs, prediction for the individual in Question 13, and a CSV of
actual versus predicted test results.

ACADEMIC NOTE
A confusion matrix is not appropriate for Linear Regression because charges is a
continuous numerical target. The notebook uses a residual plot instead.
