import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the pipe-delimited text file
df = pd.read_csv('data/MachineLearningRating_v3.txt', sep='|')
print("Data Info:")
print(df.info())
print("\nDescriptive Stats:")
print(df.describe())

# Data Quality Check
print("\nMissing Values:")
print(df.isnull().sum())

# Ensure required columns exist (adjust based on actual column names from df.info())
required_columns = ['TotalClaims', 'TotalPremium', 'Province', 'Gender', 'VehicleType', 'TransactionMonth']
available_columns = [col for col in required_columns if col in df.columns]
print("\nAvailable Columns for Analysis:", available_columns)

if all(col in df.columns for col in ['TotalClaims', 'TotalPremium']):
    # Convert TransactionMonth to datetime for temporal analysis
    df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')

    # Univariate Analysis
    plt.figure(figsize=(10, 6))
    sns.histplot(df['TotalClaims'].dropna(), bins=30, kde=True)  # Drop NA for plotting
    plt.title('Distribution of Total Claims')
    plt.savefig('plots/total_claims_dist.png')
    plt.close()

    if 'Gender' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.countplot(x='Gender', data=df)
        plt.title('Gender Distribution')
        plt.savefig('plots/gender_dist.png')
        plt.close()

    # Bivariate Analysis
    if all(col in df.columns for col in ['TotalPremium', 'TotalClaims', 'Province']):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='TotalPremium', y='TotalClaims', hue='Province', data=df)
        plt.title('Total Premium vs Total Claims by Province')
        plt.savefig('plots/premium_vs_claims.png')
        plt.close()

        correlation_matrix = df[['TotalPremium', 'TotalClaims', 'CalculatedPremiumPerTerm']].corr()
        plt.figure(figsize=(10, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.savefig('plots/correlation_matrix.png')
        plt.close()

    # Outlier Detection
    if 'TotalClaims' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(y='TotalClaims', data=df)
        plt.title('Box Plot of Total Claims')
        plt.savefig('plots/total_claims_box.png')
        plt.close()

    # Loss Ratio Calculation
    if all(col in df.columns for col in ['TotalClaims', 'TotalPremium']):
        df['LossRatio'] = df['TotalClaims'] / df['TotalPremium'].replace(0, np.nan)  # Avoid division by zero
        loss_ratio_by_province = df.groupby('Province')['LossRatio'].mean()
        print("\nLoss Ratio by Province:")
        print(loss_ratio_by_province)

        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Province', y='LossRatio', data=df)
        plt.title('Loss Ratio by Province')
        plt.xticks(rotation=45)
        plt.savefig('plots/loss_ratio_by_province.png')
        plt.close()

    # Creative Visualizations
    if 'VehicleType' in df.columns and 'TotalClaims' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.barplot(x='VehicleType', y='TotalClaims', data=df, estimator=np.mean)
        plt.title('Average Claims by Vehicle Type')
        plt.xticks(rotation=45)
        plt.savefig('plots/claims_by_vehicle_type.png')
        plt.close()

    if 'TransactionMonth' in df.columns and 'TotalClaims' in df.columns:
        plt.figure(figsize=(10, 6))
        df.groupby(df['TransactionMonth'].dt.to_period('M'))['TotalClaims'].mean().plot()
        plt.title('Temporal Trend in Claims')
        plt.savefig('plots/claims_trend.png')
        plt.close()
else:
    print("Required columns (TotalClaims, TotalPremium) not found. Check data structure.")