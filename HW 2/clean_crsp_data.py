import pandas as pd
import numpy as np

# Read the CRSP data
print("Reading CRSP data...")
crsp = pd.read_csv('crsp_1926_2020.csv')

# Display basic information about the data
print("\nInitial data shape:", crsp.shape)
print("\nColumns in the dataset:")
print(crsp.columns.tolist())
print("\nSample of the data:")
print(crsp.head())

# Apply cleaning conditions
print("\nCleaning data...")

# 1. Only include ordinary/common shares (SHRCD = 10 or 11)
crsp = crsp[crsp['SHRCD'].isin([10, 11])]

# 2. Only include stocks listed on NYSE, AMEX, or NASDAQ (EXCHCD = 1, 2, 3)
crsp = crsp[crsp['EXCHCD'].isin([1, 2, 3])]

# 3. Set negative prices to NA
crsp.loc[crsp['PRC'] < 0, 'PRC'] = np.nan

# Display cleaned data information
print("\nCleaned data shape:", crsp.shape)
print("\nNumber of missing prices after cleaning:", crsp['PRC'].isna().sum())

# Save cleaned data
print("\nSaving cleaned data...")
crsp.to_csv('crsp_cleaned.csv', index=False)
print("Done!") 