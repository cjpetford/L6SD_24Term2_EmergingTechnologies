import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
import joblib
import tkinter as tk
from tkinter import ttk

# Path to Excel file
excel_file_path = r"C:\\Users\\icego\\Desktop\\Techtorium\\Second Year\\Assessments\\Term 2 24'\\AI\\Net_Worth_Data.xlsx"

# Read Excel file into a DataFrame
df = pd.read_excel(excel_file_path)

# Drop irrelevant features from the original dataset
input_features = df.drop(columns=['Client Name', 'Client e-mail', 'Profession', 'Education', 'Country', 'Net Worth'])
output_feature = df['Net Worth']

# Transform input and output datasets into percentage-based weights between 0 and 1
input_scaler = MinMaxScaler()
input_scaled = input_scaler.fit_transform(input_features)

output_scaler = MinMaxScaler()
output_scaled = output_scaler.fit_transform(output_feature.values.reshape(-1, 1))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(input_scaled, output_scaled, test_size=0.2, random_state=42)

# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1),
    "Polynomial Regression": Pipeline([
        ('poly', PolynomialFeatures(degree=2)),
        ('linear', LinearRegression())
    ]),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "SVR": SVR(kernel='rbf'),
    "KNeighbors Regressor": KNeighborsRegressor(n_neighbors=5),
    "Extra Trees Regressor": ExtraTreesRegressor(random_state=42)
}

# Train and evaluate each model
results = {}

for name, model in models.items():
    model.fit(X_train, y_train.ravel())
    predictions = model.predict(X_test)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    results[name] = rmse
    print(f"{name} RMSE: {rmse}")

# Determine the best performing model
best_model_name = min(results, key=results.get)
print(f"\nBest Model: {best_model_name} with RMSE: {results[best_model_name]}")

# Save the best model
best_model = models[best_model_name]
joblib.dump(best_model, 'best_model.pkl')
print(f"The best model {best_model_name} has been saved as 'best_model.pkl'")

# Load the best model
loaded_model = joblib.load('best_model.pkl')

# Bar chart to visualize results
model_names = list(results.keys())
rmse_values = list(results.values())
colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))

plt.figure(figsize=(12, 6))
bars = plt.bar(model_names, rmse_values, color=colors)
plt.xlabel('Model Used')
plt.ylabel('RMSE')
plt.title('Comparison of Regression Models')

# Add RMSE values on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, yval, round(yval, 4), ha='center', va='bottom')

plt.xticks(rotation=45)
plt.show()

# GUI for user input and predictions
def predict_net_worth():
    global user_inputs_scaled, output_scaler, loaded_model
    
    # Gather user inputs
    user_inputs = pd.DataFrame({
        'Gender': [int(gender_var.get())],
        'Age': [int(age_var.get())],
        'Income': [float(income_var.get())],
        'Credit Card Debt': [float(debt_var.get())],
        'Healthcare Cost': [float(healthcare_var.get())],
        'Inherited Amount': [float(inherited_var.get())],
        'Stocks': [float(stocks_var.get())],
        'Bonds': [float(bonds_var.get())],
        'Mutual Funds': [float(mutual_funds_var.get())],
        'ETFs': [float(etfs_var.get())],
        'REITs': [float(reits_var.get())]
    })
    
    # Transform user inputs using the same scaler
    user_inputs_scaled = input_scaler.transform(user_inputs)

    # Predict net worth
    custom_predictions_scaled = loaded_model.predict(user_inputs_scaled)
    custom_predictions = output_scaler.inverse_transform(custom_predictions_scaled.reshape(-1, 1))
    
    # Display prediction
    result_var.set(f"Predicted Net Worth: ${custom_predictions[0][0]:,.2f}")

# Create the main window
root = tk.Tk()
root.title("Net Worth Predictor")

# Create input fields
gender_var = tk.StringVar()
age_var = tk.StringVar()
income_var = tk.StringVar()
debt_var = tk.StringVar()
healthcare_var = tk.StringVar()
inherited_var = tk.StringVar()
stocks_var = tk.StringVar()
bonds_var = tk.StringVar()
mutual_funds_var = tk.StringVar()
etfs_var = tk.StringVar()
reits_var = tk.StringVar()
result_var = tk.StringVar()

# Arrange input fields and labels in a grid
ttk.Label(root, text="Gender (0=Female, 1=Male):").grid(row=0, column=0, padx=10, pady=5)
ttk.Entry(root, textvariable=gender_var).grid(row=0, column=1, padx=10, pady=5)

ttk.Label(root, text="Age:").grid(row=1, column=0, padx=10, pady=5)
ttk.Entry(root, textvariable=age_var).grid(row=1, column=1, padx=10, pady=5)

ttk.Label(root, text="Income:").grid(row=2, column=0, padx=10, pady=5)
ttk.Entry(root, textvariable=income_var).grid(row=2, column=1, padx=10, pady=5)

ttk.Label(root, text="Credit Card Debt:").grid(row=3, column=0, padx=10, pady=5)
ttk.Entry(root, textvariable=debt_var).grid(row=3, column=1, padx=10, pady=5)

ttk.Label(root, text="Healthcare Cost:").grid(row=4, column=0, padx=10, pady=5)
ttk.Entry(root, textvariable=healthcare_var).grid(row=4, column=1, padx=10, pady=5)

ttk.Label(root, text="Inherited Amount:").grid(row=5, column=0, padx=10, pady=5)
ttk.Entry(root, textvariable=inherited_var).grid(row=5, column=1, padx=10, pady=5)

ttk.Label(root, text="Stocks:").grid(row=6, column=0, padx=10, pady=5)
ttk.Entry(root, textvariable=stocks_var).grid(row=6, column=1, padx=10, pady=5)

ttk.Label(root, text="Bonds:").grid(row=7, column=0, padx=10, pady=5)
ttk.Entry(root, textvariable=bonds_var).grid(row=7, column=1, padx=10, pady=5)

ttk.Label(root, text="Mutual Funds:").grid(row=8, column=0, padx=10, pady=5)
ttk.Entry(root, textvariable=mutual_funds_var).grid(row=8, column=1, padx=10, pady=5)

ttk.Label(root, text="ETFs:").grid(row=9, column=0, padx=10, pady=5)
ttk.Entry(root, textvariable=etfs_var).grid(row=9, column=1, padx=10, pady=5)

ttk.Label(root, text="REITs:").grid(row=10, column=0, padx=10, pady=5)
ttk.Entry(root, textvariable=reits_var).grid(row=10, column=1, padx=10, pady=5)

# Predict button
ttk.Button(root, text="Predict Net Worth", command=predict_net_worth).grid(row=11, column=0, columnspan=2, pady=10)

# Result display
ttk.Label(root, textvariable=result_var, font=("Helvetica", 12)).grid(row=12, column=0, columnspan=2, pady=10)

# Run the GUI loop
root.mainloop()
