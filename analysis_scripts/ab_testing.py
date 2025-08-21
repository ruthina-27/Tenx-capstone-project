import pandas as pd
import numpy as np
from scipy import stats
import os

# Configuration
OUTPUT_DIR = "analysis_outputs/statistical_tests"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the data with low_memory=False to avoid DtypeWarning
df = pd.read_csv('data/MachineLearningRating_v3.txt', sep='|', low_memory=False)
df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')

# Calculate LossRatio to align with EDA
df['LossRatio'] = df['TotalClaims'] / df['TotalPremium'].replace(0, np.nan)  # Avoid division by zero

# Select key variables and handle missing data
df_analysis = df[['Province', 'VehicleType', 'TotalClaims', 'TotalPremium', 'LossRatio']].dropna()
print(f"Data prepared for A/B testing: Shape {df_analysis.shape}, First 5 rows:\n{df_analysis.head()}")

# Save prepared data for reproducibility
df_analysis.to_csv(os.path.join(OUTPUT_DIR, "prepared_data.csv"), index=False)

# Hypothesis 1: Claim frequency difference by Province (Gauteng vs. others)
gauteng_data = df_analysis[df_analysis['Province'] == 'Gauteng']['TotalClaims']
other_provinces_data = df_analysis[df_analysis['Province'] != 'Gauteng']['TotalClaims']

# T-test with data point check
if len(gauteng_data) > 1 and len(other_provinces_data) > 1:
    t_stat, p_value = stats.ttest_ind(gauteng_data, other_provinces_data, equal_var=False)
    results = f"T-statistic (Claim Frequency): {t_stat:.4f}, P-value: {p_value:.4f}\n"
    with open(os.path.join(OUTPUT_DIR, "province_test_results.txt"), "w") as f:
        f.write(results)
    print(results.strip())

    alpha = 0.05
    if p_value < alpha:
        print("Reject H0: Significant difference in claim frequency between Gauteng and other provinces.")
    else:
        print("Fail to reject H0: No significant difference detected.")
else:
    print("Warning: Insufficient data points for Province t-test.")

# Hypothesis 2: Claim frequency by VehicleType (Passenger Vehicle vs. others)
passenger_data = df_analysis[df_analysis['VehicleType'] == 'Passenger Vehicle']['TotalClaims']
other_vehicle_data = df_analysis[df_analysis['VehicleType'] != 'Passenger Vehicle']['TotalClaims']

if len(passenger_data) > 1 and len(other_vehicle_data) > 1:
    t_stat_vehicle, p_value_vehicle = stats.ttest_ind(passenger_data, other_vehicle_data, equal_var=False)
    results_vehicle = f"VehicleType T-statistic (Claim Frequency): {t_stat_vehicle:.4f}, P-value: {p_value_vehicle:.4f}\n"
    with open(os.path.join(OUTPUT_DIR, "vehicle_type_test_results.txt"), "w") as f:
        f.write(results_vehicle)
    print(results_vehicle.strip())

    if p_value_vehicle < alpha:
        print("Reject H0: Significant difference in claim frequency between Passenger Vehicles and others.")
    else:
        print("Fail to reject H0: No significant difference detected.")
else:
    print("Warning: Insufficient data points for VehicleType t-test.")

# Hypothesis 3: LossRatio difference by Province (Gauteng vs. others)
gauteng_loss = df_analysis[df_analysis['Province'] == 'Gauteng']['LossRatio'].dropna()
other_loss = df_analysis[df_analysis['Province'] != 'Gauteng']['LossRatio'].dropna()

if len(gauteng_loss) > 1 and len(other_loss) > 1:
    t_stat_loss, p_value_loss = stats.ttest_ind(gauteng_loss, other_loss, equal_var=False)
    results_loss = f"LossRatio T-statistic: {t_stat_loss:.4f}, P-value: {p_value_loss:.4f}\n"
    with open(os.path.join(OUTPUT_DIR, "loss_ratio_test_results.txt"), "w") as f:
        f.write(results_loss)
    print(results_loss.strip())

    if p_value_loss < alpha:
        print("Reject H0: Significant difference in LossRatio between Gauteng and other provinces.")
    else:
        print("Fail to reject H0: No significant difference detected.")
else:
    print("Warning: Insufficient data points for LossRatio t-test.")