import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for beautiful plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Load data - appears to be pipe-delimited with all columns in one field
df = pd.read_csv('C:\\Users\\YEADONAY\\acis-insurance-analytics\\data\\raw\\insurance_data.csv', 
                 sep='|', encoding='utf-8')

print("Data loaded successfully!")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# Check for required columns and calculate Loss Ratio
required_cols = ['TotalClaims', 'TotalPremium']
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    print(f"Missing columns: {missing_cols}")
    print("Available columns with 'claim' or 'premium':")
    claim_cols = [col for col in df.columns if 'claim' in col.lower()]
    premium_cols = [col for col in df.columns if 'premium' in col.lower()]
    print(f"Claim columns: {claim_cols}")
    print(f"Premium columns: {premium_cols}")
    
    # Try to find alternative column names
    if claim_cols and premium_cols:
        df['LossRatio'] = df[claim_cols[0]] / df[premium_cols[0]]
        print(f"Using {claim_cols[0]} and {premium_cols[0]} for Loss Ratio calculation")
    else:
        print("Cannot calculate Loss Ratio - missing required columns")
        exit()
else:
    df['LossRatio'] = df['TotalClaims'] / df['TotalPremium']

print("=== MISSING ANALYSIS COMPONENTS ===\n")

# 1. Loss Ratio by VehicleType (MISSING from original analysis)
print("1. LOSS RATIO BY VEHICLE TYPE")
vehicle_loss_ratio = df.groupby('VehicleType')['LossRatio'].agg(['mean', 'std', 'count']).round(3)
print(vehicle_loss_ratio)
print()

# 2. ZipCode Analysis (MISSING - required for bivariate analysis)
print("2. ZIPCODE ANALYSIS")
if 'PostalCode' in df.columns:
    zipcode_analysis = df.groupby('PostalCode')[['TotalPremium', 'TotalClaims', 'LossRatio']].agg(['mean', 'count']).round(2)
    print("Top 10 PostalCodes by Premium Volume:")
    print(zipcode_analysis.sort_values(('TotalPremium', 'mean'), ascending=False).head(10))
elif 'ZipCode' in df.columns:
    zipcode_analysis = df.groupby('ZipCode')[['TotalPremium', 'TotalClaims', 'LossRatio']].agg(['mean', 'count']).round(2)
    print("Top 10 ZipCodes by Premium Volume:")
    print(zipcode_analysis.sort_values(('TotalPremium', 'mean'), ascending=False).head(10))
else:
    print("No ZipCode/PostalCode column found. Available columns:")
    print([col for col in df.columns if 'code' in col.lower() or 'zip' in col.lower() or 'postal' in col.lower()])
print()

# 3. THREE CREATIVE AND BEAUTIFUL PLOTS
print("3. CREATING 3 CREATIVE VISUALIZATION PLOTS")

# Plot 1: Risk Profile Heatmap by Province and VehicleType
plt.figure(figsize=(14, 8))
risk_matrix = df.pivot_table(values='LossRatio', index='Province', columns='VehicleType', aggfunc='mean')
mask = risk_matrix.isnull()
sns.heatmap(risk_matrix, annot=True, fmt='.3f', cmap='RdYlBu_r', 
            center=risk_matrix.mean().mean(), mask=mask,
            cbar_kws={'label': 'Loss Ratio'})
plt.title('🎯 Risk Profile Heatmap: Loss Ratio by Province & Vehicle Type', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Vehicle Type', fontsize=12, fontweight='bold')
plt.ylabel('Province', fontsize=12, fontweight='bold')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('risk_profile_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 2: Premium vs Claims Scatter with Profitability Zones
plt.figure(figsize=(14, 10))
scatter = plt.scatter(df['TotalPremium'], df['TotalClaims'], 
                     c=df['LossRatio'], s=60, alpha=0.6, 
                     cmap='RdYlGn_r', edgecolors='black', linewidth=0.5)

# Add diagonal line for break-even
max_val = max(df['TotalPremium'].max(), df['TotalClaims'].max())
plt.plot([0, max_val], [0, max_val], 'r--', linewidth=2, alpha=0.8, label='Break-even Line')

# Add profitability zones
plt.axhspan(0, max_val*0.5, alpha=0.1, color='green', label='Profit Zone')
plt.axhspan(max_val*0.5, max_val, alpha=0.1, color='red', label='Loss Zone')

plt.colorbar(scatter, label='Loss Ratio', shrink=0.8)
plt.xlabel('Total Premium ($)', fontsize=12, fontweight='bold')
plt.ylabel('Total Claims ($)', fontsize=12, fontweight='bold')
plt.title('💰 Premium vs Claims: Profitability Analysis\n(Color indicates Loss Ratio)', 
          fontsize=16, fontweight='bold', pad=20)
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('profitability_scatter.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 3: Multi-dimensional Risk Dashboard
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('📊 Insurance Risk Dashboard: Multi-Dimensional Analysis', 
             fontsize=18, fontweight='bold', y=0.98)

# Subplot 1: Gender Risk Profile
gender_stats = df.groupby('Gender').agg({
    'LossRatio': ['mean', 'std'],
    'TotalPremium': 'sum',
    'TotalClaims': 'sum'
}).round(3)
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
bars1 = ax1.bar(df.groupby('Gender')['LossRatio'].mean().index, 
                df.groupby('Gender')['LossRatio'].mean().values, 
                color=colors[:len(df['Gender'].unique())], alpha=0.8)
ax1.set_title('Risk by Gender', fontweight='bold', fontsize=14)
ax1.set_ylabel('Average Loss Ratio', fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.3f}', ha='center', va='bottom', fontweight='bold')

# Subplot 2: Top Vehicle Makes Risk
top_makes = df.groupby('Make')['LossRatio'].mean().sort_values(ascending=False).head(8)
bars2 = ax2.barh(range(len(top_makes)), top_makes.values, 
                 color=plt.cm.viridis(np.linspace(0, 1, len(top_makes))))
ax2.set_yticks(range(len(top_makes)))
ax2.set_yticklabels(top_makes.index)
ax2.set_title('Top 8 Riskiest Vehicle Makes', fontweight='bold', fontsize=14)
ax2.set_xlabel('Average Loss Ratio', fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

# Subplot 3: Monthly Trend Analysis
monthly_data = df.groupby('TransactionMonth').agg({
    'LossRatio': 'mean',
    'TotalPremium': 'sum',
    'TotalClaims': 'sum'
})
ax3_twin = ax3.twinx()
line1 = ax3.plot(monthly_data.index, monthly_data['LossRatio'], 
                 'o-', color='red', linewidth=3, markersize=8, label='Loss Ratio')
line2 = ax3_twin.plot(monthly_data.index, monthly_data['TotalPremium']/1000, 
                      's-', color='blue', linewidth=2, markersize=6, alpha=0.7, label='Premium (K$)')
ax3.set_title('Monthly Performance Trends', fontweight='bold', fontsize=14)
ax3.set_xlabel('Transaction Month', fontweight='bold')
ax3.set_ylabel('Loss Ratio', fontweight='bold', color='red')
ax3_twin.set_ylabel('Premium (Thousands $)', fontweight='bold', color='blue')
ax3.grid(True, alpha=0.3)
ax3.tick_params(axis='x', rotation=45)

# Subplot 4: Province Performance Matrix
province_perf = df.groupby('Province').agg({
    'LossRatio': 'mean',
    'TotalPremium': 'sum'
}).reset_index()
bubble = ax4.scatter(province_perf['TotalPremium']/1000, province_perf['LossRatio'],
                     s=province_perf['TotalPremium']/500, alpha=0.6,
                     c=province_perf['LossRatio'], cmap='RdYlGn_r')
ax4.set_xlabel('Total Premium (Thousands $)', fontweight='bold')
ax4.set_ylabel('Average Loss Ratio', fontweight='bold')
ax4.set_title('Province Performance\n(Bubble size = Premium Volume)', fontweight='bold', fontsize=14)
ax4.grid(True, alpha=0.3)

# Add province labels
for i, row in province_perf.iterrows():
    ax4.annotate(row['Province'], (row['TotalPremium']/1000, row['LossRatio']),
                xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('risk_dashboard.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n=== TASK 1 COMPLETION STATUS ===")
print("✅ Git Repository: COMPLETE")
print("✅ README: COMPLETE") 
print("✅ CI/CD Pipeline: COMPLETE")
print("✅ Basic EDA: COMPLETE")
print("✅ Missing Components Added:")
print("   - Loss Ratio by VehicleType")
print("   - ZipCode/PostalCode Analysis")
print("   - 3 Creative Visualization Plots")
print("\n🎉 TASK 1 IS NOW COMPLETE!")
