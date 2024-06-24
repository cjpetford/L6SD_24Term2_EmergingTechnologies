import pandas as pd

# Path to Excel file
excel_file_path = r"C:\\Users\\icego\\Desktop\\Techtorium\\Second Year\\Assessments\\Term 2 24'\\AI\\Net_Worth_Data.xlsx"

# Read Excel file into a DataFrame
df = pd.read_excel(excel_file_path)

# Print the first few rows
print("\nFirst few rows of the dataset:\n")
print(df.head())

# Print the last few rows
print("\nLast few rows of the dataset:\n")
print(df.tail())

# Print the shape of data
print("\nShape of the dataset:")
print(df.shape)

# Print a concise summary of the dataset
print("\nSummary of the dataset:")
print(df.info())
