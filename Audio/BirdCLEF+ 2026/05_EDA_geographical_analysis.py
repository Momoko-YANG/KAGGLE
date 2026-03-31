import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set professional style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (20, 12)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['lines.linewidth'] = 2

print("="*80)
print("GEOSPATIAL ANALYSIS OF BIRD RECORDINGS")
print("="*80)

# Check if necessary columns exist
required_cols = ['latitude', 'longitude', 'class_name']
missing_cols = [col for col in required_cols if col not in train_df.columns]

if missing_cols:
    print(f"\n❌ Missing required columns: {missing_cols}")
    print(f"Available columns: {train_df.columns.tolist()}")

    # Try to identify latitude/longitude columns if they have different names
    potential_lat = [col for col in train_df.columns if 'lat' in col.lower()]
    potential_lon = [col for col in train_df.columns if 'lon' in col.lower() or 'long' in col.lower()]

    if potential_lat and potential_lon:
        print(f"\n✅ Found potential coordinate columns:")
        print(f"   Latitude: {potential_lat[0]}")
        print(f"   Longitude: {potential_lon[0]}")

        # Rename columns for consistency
        train_df = train_df.rename(columns={
            potential_lat[0]: 'latitude',
            potential_lon[0]: 'longitude'
        })

        print("   Columns renamed successfully!")
    else:
        print("\n❌ Cannot identify coordinate columns. Please ensure your data contains latitude and longitude columns.")
        raise ValueError("Missing coordinate columns")
    
# Check for class_name column
if 'class_name' not in train_df.columns:
    if 'primary_label' in train_df.columns:
        train_df['class_name'] = train_df['primary_label']
        print("✅ Using 'primary_label' as class_name")
    elif 'species' in train_df.columns:
        train_df['class_name'] = train_df['species']
        print("✅ Using 'species' as class_name")
    else:
        print("❌ No species/class column found. Creating dummy class for visualization.")
        train_df['class_name'] = 'All Species'

# Define color sheme based on unique classes
unique_classes = train_df['class_name'].unique()[:10]  # Limit to top 10 for clarity
color_palette = sns.color_palette("husl", len(unique_classes))
CLASS_COLORS = {cls: color_palette[i] for i, cls in enumerate(unique_classes)}

print(f"\n📊 Data Overview:")
print(f"   Total recordings: {len(train_df):,}")
print(f"   Unique species: {train_df['class_name'].nunique()}")
print(f"   Latitude range: [{train_df['latitude'].min():.2f}°, {train_df['latitude'].max():.2f}°]")
print(f"   Longitude range: [{train_df['longitude'].min():.2f}°, {train_df['longitude'].max():.2f}°]")

# ============================================================================
# CREATE MAIN GEOGRAPHIC VISUALIZATION
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(20, 10))
fig.suptitle('Geospatial Analysis of Bird Species Recordings', fontsize=18, fontweight='bold', y=0.98)

# 1. GLOBAL VIEW - Worldwide Distribution
ax1 = axes[0]

# Plot each species class with transparency
for cls, color in CLASS_COLORS.items():
    mask = train_df['class_name'] == cls
    if mask.sum() > 0: 
        ax1.scatter(train_df.loc[mask, 'longitude'],
                    train_df.loc[mask, 'latitude'],
                    c=[color], s=5, alpha=0.4, label=cls,
                    rasterized=True, edgecolors='none')

ax1.set_xlabel("Longitude", fontsize=12, fontweight='semibold')
ax1.set_ylabel("Latitude", fontsize=12, fontweight='semibold')
ax1.set_title("Global Distribution of Bird Recordings", fontsize=14, fontweight='bold', pad=15)

# Add grid lines at major intervals
ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.2, linewidth=0.5)
ax1.axvline(x=0, color='gray', linestyle='-', alpha=0.2, linewidth=0.5)

# Set reasonable limits based on data
ax1.set_xlim(train_df['longitude'].min() - 10, train_df['longitude'].max() + 10)
ax1.set_ylim(train_df['latitude'].min() - 10, train_df['latitude'].max() + 10)

# Customize legend
if len(unique_classes) <= 15:
    legend1 = ax1.legend(markerscale=2, framealpha=0.9, fontsize=9,
                         loc='upper left', bbox_to_anchor=(1.01, 1))
    legend1.get_frame().set_facecolor('white')
    legend1.get_frame().set_edgecolor('gray')

# Add statistical annotation
stats_text = f"Total Recordings: {len(train_df):,}\nUnique Species: {train_df['class_name'].nunique()}"
ax1.text(0.02, 0.02, stats_text, transform=ax1.trans)
