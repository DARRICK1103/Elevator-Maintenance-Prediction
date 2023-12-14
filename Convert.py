import pandas as pd

# Read the CSV file
data = pd.read_csv('predictive-maintenance-dataset.csv')

# Replace NaN values with previous row's values
data.fillna(method='ffill', inplace=True)

# Export the modified data to a new CSV file
data.to_csv('updated_file.csv', index=False)

print("NaN values replaced and saved to updated_file.csv")