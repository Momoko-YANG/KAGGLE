import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from scipy import stats
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

# Set professional style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")
plt.rcParams['figure.figsize'] = (20, 14)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['lines.linewidth'] = 2

print("="*80)
print("BIRD SPECIES DISTRIBUTION ANALYSIS - COMPREHENSIVE REPORT")
print("="*80)

# Create figure with enhanced subplots
fig = plt.figure(figsize=(20, 14))
fig.suptitle('Bird Species Distraibution Analysis', fontsize=20, fontweight='bold', y=0.98)

# 1. Top 30 Species Bar Chart(Enhanced)
ax1 = plt.subplot(2, 2, 1)
species_counts = train_df['primary_label'].value_counts()
top_30_species = species_counts.head(30)

# Create gradient colors
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(top_30_species)))
bars = ax1.barh(range(len(top_30_species)), top_30_species.values,
                colors=colors, edgecolor='white', linewidth=0.5)

# Customize axes
ax1.set_yticks(range(len(top_30_species)))
ax1.set_yticklabels(top_30_species.index, fontsize=9)
ax1.set_xlabel('Number of Recordings', fontsize=12, fontweight='semibold')
ax1.set_ylabel('Bird Species', fontsize=12, fontweight='semibold')
ax1.set_title('Top 30 Most Common Bird Species', fontsize=14, fontweight='bold', pad=15)
ax1.invert_yaxis()
ax1.grid(axis='x', aplha=0.3)

# Add value lables with better formatting
for i, (bar, value) in enumerate(zip(bars, top_30_species.values)):
    ax1.text(value, bar.get_y() + bar.get_height()/2,
             f'{value:,}', ha='left', va='center',
             fontsize=8, fontweight='bold', color='black')
    
# Add total count annotation
ax1.text(0.98, 0.02, f'Total: {top_30_species.sum():,} recordings \n({print("="*80), 
                                                                      print("BIRD SPECIES DISTRIBUTION ANALYSIS - COMPREHENSIVE REPORT"), 
                                                                      print("="*80)}%)',
    transform=ax1.transAxes, fontsize=9, ha='right',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 2. Enhanced Statstics Panel with Distribution Metrics
ax2 = plt.subplot(2, 2, 2)
ax2.axis('off')

# Calculate additional statistics
gini_coefficient = 1 - np.sum((species_counts.values / species_counts.sum())**2)
skewness = stats.skew(species_counts.values)
kurt = stats.kurtosis(species_counts.values)

# Create enhanced statistics table
stats_text = """
╔══════════════════════════════════════════════════════════════╗
║           SPECIES DISTRIBUTION STATISTICS                    ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  📊 Basic Statistics:                                        ║
║  ┌────────────────────────────────────────────────────────┐  ║
║  │ Mean samples per species:      {mean:>10,.1f}          │  ║
║  │ Median samples per species:    {median:>10,.1f}        │  ║
║  │ Standard deviation:            {std:>10,.2f}           │  ║
║  │ Total species:                {total:>10,}             │  ║
║  │ Total recordings:             {total_rec:>10,}         │  ║
║  └────────────────────────────────────────────────────────┘  ║
║                                                              ║
║  ⚠️  Class Imbalance:                                        ║
║  ┌────────────────────────────────────────────────────────┐  ║
║  │ Minimum samples: {min_samples:>6,} ({min_species:<12}) │  ║
║  │ Maximum samples: {max_samples:>6,} ({max_species:<12}) │  ║
║  │ Imbalance ratio:     {imbalance_ratio:>10.2f}:1        │  ║
║  │ Gini coefficient:             {gini:>10.4f}            │  ║
║  └────────────────────────────────────────────────────────┘  ║
║                                                              ║
║  🔬 Rare Species Analysis:                                   ║
║  ┌────────────────────────────────────────────────────────┐  ║
║  │ Threshold:                  < {rare_thresh} recordings │  ║
║  │ Rare species count:                {rare_count:>10,}   │  ║
║  │ Percentage of total:               {rare_pct:>10.2f}%  │  ║
║  └────────────────────────────────────────────────────────┘  ║
║                                                              ║
║  📈 Distribution Shape:                                      ║
║  ┌────────────────────────────────────────────────────────┐  ║
║  │ Skewness:                         {skewness:>10.3f}    │  ║
║  │ Kurtosis:                         {kurtosis:>10.3f}    │  ║
║  └────────────────────────────────────────────────────────┘  ║
╚══════════════════════════════════════════════════════════════╝
""".format(
    mean=species_counts.mean(),
    median=species_counts.median(),
    std=species_counts.std(),
    total=len(species_counts),
    total_rec=species_counts.sum(),
    min_samples=species_counts.min(),
    min_species=species_counts.idxmin()[:12],
    max_samples=species_counts.max(),
    max_species=species_counts.idxmax()[:12],
    imbalance_ratio=species_counts.max() / species_counts.min(),
    gini=gini_coefficient,
    rare_thresh=10,
    rare_count=len(species_counts[species_counts < 10]),
    rare_pct=(len(species_counts[species_counts < 10]) / len(species_counts)) * 100,
    skewness=skewness,
    kurtosis=kurt
)

ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#f8f9fa', alpha=0.95, edgecolor='gray'))

# 3. Enhanced Distribution Histogram with KDE
ax3 = plt.subplot(2, 2, 3)
log_counts = np.log10(species_counts.values + 1)

# Plot Histogram
counts, bins, patches = ax3.hist(log_counts, bins=50, color='#2c7fb8',
                                 alpha=0.7, edgecolor='black', linewidth=0.5,
                                 density=True, label='Histogram')

# Add KDE
kde = gaussian_kde(log_counts)
x_range = np.linspace(log_counts.min(), log_counts.max(), 200)
ax3.plot(x_range, kde(x_range), color='#F18F01', linewidth=2.5, label='KDE')

# Add mean and median lines
median_log = np.median(log_counts)
mean_log = np.mean(log_counts)
ax3.axvline(median_log, color='red', linestyle='--', linewidth=2, 
            label=f'Median: {species_counts.median():.0f} recordings')
ax3.axvline(mean_log, color='orange', linestyle='--', linewidth=2, 
            label=f'Mean: {species_counts.mean():.1f} recordings')

ax3.set_xlabel('Log10(Number of Recordings + 1)', fontsize=11, fontweight='semibold')
ax3.set_ylabel('Density', fontsize=11, fontweight='semibold')
ax3.set_title('Species Frequency Distribution (Log Scale with KDE)',
              fontsize=13, fontweight='bold', pad=15)
ax3.legend(loc='upper right', fontsize=9, framealpha=0.9)
ax3.grid(alpha=0.3)

# Add annotation for rare species
rare_count = len(species_counts[species_counts < 10])
x_rare = np.log10(10)
y_rare = kde(x_rare)[0] if x_rare > log_counts.min() and x_rare < log_counts.max() else 0.5
ax3.annotate(f'{rare_count} rare species\n(< 10 recordings)', 
             xy=(x_rare, y_rare), xytext=(x_rare + 0.3, y_rare + 0.5),
             arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
             fontsize=9, ha='center', 
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

# 4. Enhanced Cumulative Distribution
ax4 = plt.subplot(2, 2, 4)
sorted_counts = species_counts.sort_values(ascending=False)
cumulative_pct = np.cumsum(sorted_counts.values) / sorted_counts.sum() * 100
species_pct = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts) * 100

# Plot main line
ax4.plot(species_pct, cumulative_pct, linewidth=3, color='#2c7fb8', label='Cumulative Distribution')
ax4.fill_between(species_pct, cumulative_pct, alpha=0.3, color='#2c7fb8')

# Add reference lines
ax4.axhline(y=80, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='80% threshold')
ax4.axhline(y=50, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='50% threshold')

# Find points of interest
species_80_idx = np.where(cumulative_pct >= 80)[0][0] if len(np.where(cumulative_pct >= 80)[0]) > 0 else len(sorted_counts) - 1
species_50_idx = np.where(cumulative_pct >= 50)[0][0] if len(np.where(cumulative_pct >= 50)[0]) > 0 else len(sorted_counts) - 1
species_80 = species_80_idx + 1
species_50 = species_50_idx + 1

ax4.axvline(x=(species_80/len(sorted_counts))*100, color='red', linestyle='--',  linewidth=1.5, alpha=0.7)
ax4.axvline(x=(species_50/len(sorted_counts))*100, color='orange', linestyle='--',  linewidth=1.5, alpha=0.7)

# Add scatter points at key positions
ax4.plot((species_80/len(sorted_counts))*100, 80, 'ro', markersize=8, markeredgecolor='white')
ax4.plot((species_50/len(sorted_counts))*100, 50, 'o', color='orange', markersize=8, markeredgecolor='white')

ax4.set_xlabel('Percentage of Species', fontsize=11, fontweight='semibold')
ax4.set_ylabel('Cumulative Percentage of Recordings', fontsize=11, fontweight='semibold')
ax4.set_title('Cumulative Distribution: Species vs. Recordings', fontsize=13, fontweight='bold', pad=15)
ax4.grid(alpha=0.3)
ax4.set_xlim(0, 100)
ax4.set_ylim(0, 100)
ax4.legend(loc='lower right', fontsize=9)

# Add enhanced annotations
bbox_props = dict(boxstyle='round, pad=0.3', facecolor='white', alpha=0.8)
ax4.annotate(f'Top {species_80} species\n({(species_80/len(sorted_counts))*100:.1f}%)\naccount for 80%', 
             xy=((species_80/len(sorted_counts))*100, 80), 
             xytext=((species_80/len(sorted_counts))*100 + 10, 75),
             arrowprops=dict(arrowstyle='->', color='red', lw=1),
             fontsize=8, bbox=bbox_props)

ax4.annotate(f'Top {species_50} species\n({(species_50/len(sorted_counts))*100:.1f}%)\naccount for 50%', 
             xy=((species_50/len(sorted_counts))*100, 50), 
             xytext=((species_50/len(sorted_counts))*100 + 10, 45),
             arrowprops=dict(arrowstyle='->', color='orange', lw=1),
             fontsize=8, bbox=bbox_props)

# Add enhanced insights panel
insight_text = f"""
📈 KEY INSIGHTS:
• Most common: {species_counts.idxmax()} ({species_counts.max():,} recordings, {species_counts.max()/species_counts.sum()*100:.1f}%)
• Rarest: {species_counts.idxmin()} ({species_counts.min()} recordings)
• Top 10% species: {cumulative_pct[int(0.1*len(sorted_counts))]:.1f}% of recordings
• Bottom 50% species: {100 - cumulative_pct[len(sorted_counts)//2]:.1f}% of recordings
• Diversity index (Simpson): {gini_coefficient:.4f}
"""

ax4.text(0.02, 0.02, insight_text, transform=ax4.transAxes, fontsize=8,
         verticalalignment='bottom', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

plt.tight_layout()
plt.subplots_adjust(top=0.95, hspace=0.3, wspace=0.25)

# Save the figure
plt.savefig('bird_species_distribution_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# Create additional visualization: Species Distribution Pie Chart
fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))
fig2.suptitle('Species Distribution Overview', fontsize=16, fontweight='bold')

# 1. Species Categories Pie Chart
ax_pie = axes2[0]
categories = {
    'Rare (<10)': len(species_counts[species_counts < 10]),
    'Low (10-50)': len(species_counts[(species_counts >= 10) & (species_counts < 50)]),
    'Medium (50-200)': len(species_counts[(species_counts >= 50) & (species_counts < 200)]),
    'High (200-1000)': len(species_counts[(species_counts >= 200) & (species_counts < 1000)]),
    'Very High (>1000)': len(species_counts[species_counts >= 1000])
}

colors_pie = ['#A23B72', '#F18F01', '#2E86AB', '#6A994E', '#C73E1D']
wedges, texts, autotexts = ax_pie.pie(categories.values(), labels=categories.keys(), 
                                        colors=colors_pie, autopct='%1.1f%%', 
                                        startangle=90, explode=[0.05]*len(categories))
ax_pie.set_title('Species Distribution by Frequency Category', fontsize=12, fontweight='bold')

# 2. Top 10 vs Bottom 10 Comparison
ax_bar = axes2[1]
top_10 = species_counts.head(10)
bottom_10 = species_counts.tail(10)

x = np.arange(10)
width = 0.35

bars1 = ax_bar.bar(x - width/2, top_10.values, width, label='Top 10 Species', 
                   color='#2E86AB', alpha=0.7, edgecolor='white')
bars2 = ax_bar.bar(x + width/2, bottom_10.values, width, label='Bottom 10 Species', 
                   color='#F18F01', alpha=0.7, edgecolor='white')

ax_bar.set_xlabel('Species Rank', fontsize=11)
ax_bar.set_ylabel('Number of Recordings', fontsize=11)
ax_bar.set_title('Top 10 vs Bottom 10 Species Comparison', fontsize=12, fontweight='bold')
ax_bar.set_xticks(x)
ax_bar.set_xticklabels([f'#{i+1}' for i in range(10)], fontsize=9)
ax_bar.legend()
ax_bar.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax_bar.text(bar.get_x() + bar.get_width()/2., height + max(top_10.values)*0.01,
                       f'{int(height)}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('species_distribution_categories.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# Print enhanced summary statistics in console
print("\n" + "="*80)
print("SPECIES DISTRIBUTION ANALYSIS - COMPLETE SUMMARY")
print("="*80)
print(f"\n📊 OVERALL STATISTICS:")
print(f"   • Total species: {len(species_counts):,}")
print(f"   • Total recordings: {species_counts.sum():,}")
print(f"   • Unique species: {species_counts.nunique()}")

print(f"\n📈 DISTRIBUTION METRICS:")
print(f"   • Mean: {species_counts.mean():.2f} recordings/species")
print(f"   • Median: {species_counts.median():.2f} recordings/species")
print(f"   • Standard Deviation: {species_counts.std():.2f}")
print(f"   • Variance: {species_counts.var():.2f}")
print(f"   • Skewness: {skewness:.3f} (Highly right-skewed)")
print(f"   • Kurtosis: {kurt:.3f} (Heavy-tailed distribution)")

print(f"\n⚠️  CLASS IMBALANCE METRICS:")
print(f"   • Minimum: {species_counts.min()} ({species_counts.idxmin()})")
print(f"   • Maximum: {species_counts.max():,} ({species_counts.idxmax()})")
print(f"   • Imbalance ratio: {species_counts.max()/species_counts.min():.2f}:1")
print(f"   • Gini coefficient: {gini_coefficient:.4f} (0=perfect equality, 1=perfect inequality)")

print(f"\n🔬 RARE SPECIES ANALYSIS (<10 recordings):")
rare_species = species_counts[species_counts < 10]
print(f"   • Count: {len(rare_species)} species")
print(f"   • Percentage: {(len(rare_species)/len(species_counts))*100:.2f}%")
print(f"   • Total recordings from rare species: {rare_species.sum():,} ({(rare_species.sum()/species_counts.sum())*100:.2f}%)")
if len(rare_species) > 0:
    print(f"   • Examples: {', '.join(rare_species.head(5).index.tolist())}")

print(f"\n📊 CONCENTRATION METRICS:")
print(f"   • Top 1 species: {species_counts.max()/species_counts.sum()*100:.1f}% of all recordings")
print(f"   • Top 10 species: {top_10.sum()/species_counts.sum()*100:.1f}% of all recordings")
print(f"   • Top 50 species: {species_counts.head(50).sum()/species_counts.sum()*100:.1f}% of all recordings")
print(f"   • Bottom 50% species: {species_counts.tail(len(species_counts)//2).sum()/species_counts.sum()*100:.1f}% of all recordings")

print(f"\n🎯 RECOMMENDATIONS FOR MODEL TRAINING:")
print(f"   • Use weighted loss functions to handle class imbalance")
print(f"   • Consider oversampling rare classes or using data augmentation")
print(f"   • Focus on top {species_80} species for initial model development")
print(f"   • Implement stratified sampling for train/validation splits")
print(f"   • Consider removing or grouping species with <5 samples")

print("\n" + "="*80)
print("✅ Analysis complete! High-resolution images saved:")
print("   • bird_species_distribution_analysis.png")
print("   • species_distribution_categories.png")
print("="*80)