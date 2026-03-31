import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from scipy import stats
from scipy.stats import gaussian_kde
import librosa
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set professional style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")
plt.rcParams['figure.figsize'] = (20, 12)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['lines.linewidth'] = 2

print("="*80)
print("AUDIO DURATION ANALYSIS - COMPREHENSIVE REPORT")
print("="*80)

def get_audio_info(filepath):
    """Extract audio duration and sample rate"""
    try:
        info = librosa.get_duration(path=filepath)
        return info
    except:
        return np.nan
    
# Analyze audio durations (sample 2000 files for efficiency)
sample_size = min(2000, len(train_df))
sample_files = train_df['filename'].sample(sample_size, random_state=42)

durations = []
failed_files = []

print("\n Analyzing audio files...")
for filename in tqdm(sample_files, desc="Analyzing audio durations"):
    filepath = TRAIN_AUDIO_PATH / filename
    if filepath.exists():
        duration = get_audio_info(filepath)
        if not np.isnan(duration):
            durations.append(duration)
        else:
            failed_files.append(filename)
    else:
        failed_files.append(filename)

# Convert to numpy array for efficient computation
durations = np.array(durations)

# Creat professional figure with enhanced subplots
fig = plt.figure(figsize=(20, 14))
fig.suptitle('Audio Duration Analysis', fontsize=20, fontweight='bold', y=0.98)

# Color scheme
primary_color = '#2E86AB'
secondary_color = '#F18F01'
accent_color = '#A23B72'

# 1. Histogram with KDE (Top Left)
ax1 = plt.subplot(2, 3, 1)
counts, bins, patches = ax1.hist(durations, bins=50, alpha=0.65, color=primary_color,
                                 edgecolor='white', linewidth=0.5, density=True, label='Histogram')

# Add KDE
kde = gaussian_kde(durations)
x_range = np.linspace(durations.min(), durations.max(), 200)
ax1.plot(x_range, kde(x_range), color=secondary_color, linewidth=2.5, label='KDE')

# Add mean and median lines
mean_val = np.mean(durations)
median_val = np.median(durations)
ax1.axvline(mean_val, color='red', linestyle='--', linewidth=2,
            label=f'Mean: {mean_val:.2f}s', alpha=0.8)
ax1.axvline(median_val, color='green', linestyle='--', linewidth=2,
            label=f'Median: {median_val:.2f}s', alpha=0.8)

ax1.set_xlabel('Duration (seconds)', fontsize=12, fontweight='semibold')
ax1.set_ylabel('Density', fontsize=12, fontweight='semibold')
ax1.set_title('Duration Distribution with KDE', fontsize=13, fonweight='bold', pad=15)
ax1.legend(loc='upper right', fontsize=10, framealpha=0.9)
ax1.grid(alpha=0.3)
ax1.set_facecolor('#f8f9fa')

# Add statistical annotation
ax1.text(0.02, 0.95, f'Total samples: {len(durations):,}\nFailed: {len(failed_files):,}', 
         transform=ax1.transAxes, fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 2. Enhanced Box Plot with Statistics (Top Middle)
ax2 = plt.subplot(2, 3, 2)
bp = ax2.boxplot(durations, vert=True, patch_artist=True, widths=0.6, showmeans=True,
                 meanline=True, meanprops=dict(color='blue', linestyle='--', linewidth=2))

# Customize boxplot
bp['boxes'][0].set_facecolor(primary_color)
bp['boxes'][0].set_alpha(0.7)
bp['boxes'][0].set_edgecolor('black')
bp['whiskers'][0].set_color('black')
bp['whiskers'][1].set_color('black')
bp['caps'][0].set_color('black')
bp['caps'][1].set_color('black')
bp['medians'][0].set_color('red')
bp['medians'][0].set_linewidth(2)
bp['means'][0].set_color('blue')
bp['means'][0].set_linewidth(2)
bp['fliers'][0].set_markerfacecolor(accent_color)
bp['fliers'][0].set_markeredgecolor(accent_color)
bp['fliers'][0].set_alpha(0.5)
bp['fliers'][0].set_markersize(4)

ax2.set_ylabel('Durantion (seconds)', fontsize=12, fontweight='semibold')
ax2.set_title('Durantion Distribution - Box Plot', fontsize=13, fontweight='bold', pad=15)
ax2.set_xticklabels(['All Recordings'], fontsize=11)
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_facecolor('#f8f9fa')

# Add statistical annotations
q1 = np.percentile(durations, 25)
q3 = np.percentile(durations, 75)
iqr = q3 - q1
stats_text = f"Q1: {q1:.2f}s\nQ3: {q3:.2f}s\nIQR: {iqr:.2f}s\nMean: {mean_val:.2f}s"
ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes, fontsize=9,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


# 3. Cumulative Distribution with Percentiles (Top Right)
ax3 = plt.subplot(2, 3, 3)
sorted_durations = np.sort(durations)
cumulative = np.arange(1, len(sorted_durations) + 1) / len(sorted_durations)

ax3.plot(sorted_durations, cumulative, linewidth=3, color=secondary_color, label='CDF')
ax3.fill_between(sorted_durations, cumulative, alpha=0.3, color=secondary_color)

# Add percentile markers
percentiles = [10, 25, 50, 75, 90]
percentile_values = np.percentile(durations, percentiles)
colors_percentile = ['gray', 'blue', 'red', 'orange', 'purple']

for p, val, color in zip(percentiles, percentile_values, colors_percentile):
    ax3.axhline(y=p/100, color=color, linestyle=':', alpha=0.5, linewidth=1)
    ax3.axvline(x=val, color=color, linestyle=':', alpha=0.5, linewidth=1)
    ax3.plot(val, p/100, 'o', color=color, markersize=6, markeredgecolor='white')
    ax3.annotate(f'{p}th: {val:.1f}s', xy=(val, p/100), xytext=(val + 0.5, p/100 - 0.05),
                 fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

ax3.set_xlabel('Duration (seconds)', fontsize=12, fontweight='semibold')
ax3.set_ylabel('Cumulative Probability', fontsize=12, fontweight='semibold')
ax3.set_title('Cumulative Distribution Function', fontsize=13, fontweight='bold', pad=15)
ax3.grid(True, alpha=0.3)
ax3.legend(loc='lower right', fontsize=10)
ax3.set_facecolor('#f8f9fa')

# 4. Statistical Summary Card (Bottom Left)
ax4 = plt.subplot(2, 3, 4)
ax4.axis('off')

# Calculate additional statistics
mode_val = stats.mode(durations, keepdims=True)[0][0] if len(durations) > 0 else 0
skewness = stats.skew(durations)
kurtosis = stats.kurtosis(durations)

# Create enhanced statistics table
stats_summary = f"""
╔══════════════════════════════════════════════════════════════╗
║                 AUDIO DURATION STATISTICS                    ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  📊 Basic Statistics:                                        ║
║  ┌────────────────────────────────────────────────────────┐  ║
║  │ Total files analyzed:         {len(durations):>8,}     │  ║
║  │ Failed files:                 {len(failed_files):>8,}  │  ║
║  │ Success rate: {len(durations)/(len(durations)+len(failed_files))*100:>7.1f}%                      │  ║
║  └────────────────────────────────────────────────────────┘  ║
║                                                              ║
║  📈 Central Tendency:                                        ║
║  ┌────────────────────────────────────────────────────────┐  ║
║  │ Mean duration:               {mean_val:>8.2f} seconds  │  ║
║  │ Median duration:             {median_val:>8.2f} seconds│  ║
║  │ Mode duration:               {mode_val:>8.2f} seconds  │  ║
║  └────────────────────────────────────────────────────────┘  ║
║                                                              ║
║  📉 Dispersion Metrics:                                      ║
║  ┌────────────────────────────────────────────────────────┐  ║
║  │ Standard deviation:  {np.std(durations):>8.2f} seconds │  ║
║  │ Variance:            {np.var(durations):>8.2f} seconds²│  ║
║  │ Range:{durations.max() - durations.min():>8.2f} seconds│  ║
║  │ IQR:                         {iqr:>8.2f} seconds       │  ║
║  └────────────────────────────────────────────────────────┘  ║
║                                                              ║
║  📐 Distribution Shape:                                      ║
║  ┌────────────────────────────────────────────────────────┐  ║
║  │ Skewness:                    {skewness:>8.3f}          │  ║
║  │ Kurtosis:                    {kurtosis:>8.3f}          │  ║
║  └────────────────────────────────────────────────────────┘  ║
╚══════════════════════════════════════════════════════════════╝
"""

ax4.text(0.05, 0.95, stats_summary, transform=ax4.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.9, edgecolor='gray'))

# 5. Percentile Analysis (Bottom Middle)
ax5 = plt.subplot(2, 3, 5)
percentile_range = np.arange(0, 101, 5)
percentile_vals = np.percentile(durations, percentile_range)

ax5.plot(percentile_range, percentile_vals, markers='o', markersize=4,
         linewidth=2, color=accent_color, markerfacecolor='white', markeredgewidth=1.5)
ax5.fill_between(percentile_range, percentile_vals, alpha=0.2, color=accent_color)

ax5.set_xlabel('Percentile', fontsize=12, fontweight='semibold')
ax5.set_ylabel('Durantion (seconds)', fontsize=12, fontweight='semibold')
ax5.set_title('Percentile Analysis', fontsize=13, fontweight='bold', pad=15)
ax5.grid(True, alpha=0.3)
ax5.set_xticks(np.arange(0, 101, 10))
ax5.set_facecolor('#f8f9fa')

# Add key percentile annotations
key_percentiles = [10, 25, 50, 75, 90, 95, 99]
for p in key_percentiles:
    val = np.percentile(durations, p)
    ax5.annotate(f'{p}%: {val:.1f}s', xy=(p, val), xytext=(p + 2, val + 0.5),
                fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

# 6. Durantion Range Distribution (Bottom Right)
ax6 = plt.subplot(2, 3, 6)

# Create durantion bins
bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, np.inf]
labels = ['0-5', '5-10', '10-15', '15-20', '20-25', '25-30', 
          '30-35', '35-40', '40-45', '45-50', '50-55', '55-60', '60+']
duration_bins = pd.cut(durations, bins=bins, labels=labels, right=False)
bin_counts = duration_bins.value_counts().sort_index()

colors_bins = plt.cm.Blues(np.linspace(0.4, 0.9, len(bin_counts)))
bars = ax6.bar(range(len(bin_counts)), bin_counts.values, color=colors_bins,
               edgecolor='white', linewidth=1)

ax6.set_xticks(range(len(bin_counts)))
ax6.set_xticklabels(bin_counts.index, rotation=45, ha='right', fontsize=9)
ax6.set_xlabel('Duration Range (seconds)', fontsize=12, fontweight='semibold')
ax6.set_ylabel('Number of Recordings', fontsize=12, fontweight='semibold')
ax6.set_title('Duration Range Distribution', fontsize=13, fontweight='bold', pad=15)
ax6.grid(True, alpha=0.3, axis='y')
ax6.set_facecolor('#f8f9fa')

# Add value labels on bars
max_height = max(bin_counts.values)
for bar, val in zip(bars, bin_counts.values):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height + max_height * 0.01,
             f'{val:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
# Add percentage annotation
total_samples = len(durations)
ax6.text(0.98, 0.95, f'Total: {total_samples:,}', transform=ax6.transAxes, 
         fontsize=9, ha='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.subplots_adjust(top=0.94, hspace=0.3, wspace=0.25)

# Save high-quality image
plt.savefig('audio_durantion_analysis_professional.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.show()

# Create additional visualization: Outlier Analysis
fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))
fig2.suptitle('Outlier Analysis and Duration Patterns', fontsize=16, fontweight='bold')

# 1. Outlier Detection Plot
ax_out = axes2[0]
q1 = np.percentile(durations, 25)
q3 = np.percentile(durations, 75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outliers = durations[(durations < lower_bound) | (durations > upper_bound)]
normal_data = durations[(durations >= lower_bound) & (durations <= upper_bound)]

# Create violin plot
parts = ax_out.violinplot([normal_data], positions=[1], widths=0.7, showmeans=True, showmedians=True)
for pc in parts['bodies']:
    pc.set_facecolor(primary_color)
    pc.set_alpha(0.7)


# Add outliers as scatter points
if len(outliers) > 0:
    ax_out.scatter(np.random.normal(1, 0.04, len(outliers)), outliers,
                   color=accent_color, alpha=0.5, s=20, label=f'Outliers ({len(outliers)})')
    
ax_out.set_xticks([1])
ax_out.set_xticklabels(['All Recordings'])
ax_out.set_ylabel('Duration (seconds)', fonsize=11)
ax_out.set_title('Outlier Analysis', fontsize=12, fontweight='bold')
ax_out.grid(True, alpha=0.3, axis='y')
ax_out.legend()

# Add boundary lines
ax_out.axhline(y=lower_bound, color='red', linestyle='--', alpha=0.5, label=f'Lower bound: {lower_bound:.2f}s')
ax_out.axhline(y=upper_bound, color='red', linestyle='--', alpha=0.5, label=f'Upper bound: {upper_bound:.2f}s')


# 2. Duration vs Sample Index
ax_time = axes2[1]
ax_time.plot(range(len(durations)), durations, 'o', color=primary_color, alpha=0.3, markersize=2)
ax_time.axhline(y=mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}s')
ax_time.axhline(y=median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}s')
ax_time.fill_between(range(len(durations)), lower_bound, upper_bound, alpha=0.2, color='gray', label='IQR Range')

ax_time.set_xlabel('Sample Index', fontsize=11)
ax_time.set_ylabel('Duration (seconds)', fontsize=11)
ax_time.set_title('Duration by Sample Index', fontsize=12, fontweight='bold')
ax_time.legend(loc='upper right', fontsize=9)
ax_time.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('audio_duration_outlier_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()


# Print comprehensive statistics in console
print("\n" + "="*80)
print("AUDIO DURATION ANALYSIS - COMPREHENSIVE REPORT")
print("="*80)

print(f"\n📊 SAMPLE INFORMATION:")
print(f"   • Total files analyzed: {len(durations):,}")
print(f"   • Failed files: {len(failed_files):,}")
print(f"   • Success rate: {len(durations)/(len(durations)+len(failed_files))*100:.2f}%")

print(f"\n📈 CENTRAL TENDENCY:")
print(f"   • Mean duration: {mean_val:.2f} seconds")
print(f"   • Median duration: {median_val:.2f} seconds")
print(f"   • Mode: {mode_val:.2f} seconds")

print(f"\n📉 DISPERSION METRICS:")
print(f"   • Standard deviation: {np.std(durations):.2f} seconds")
print(f"   • Variance: {np.var(durations):.2f} seconds²")
print(f"   • Range: {durations.min():.2f} - {durations.max():.2f} seconds ({durations.max() - durations.min():.2f}s)")
print(f"   • Interquartile Range (IQR): {iqr:.2f} seconds")

print(f"\n📊 PERCENTILE ANALYSIS:")
for p in [10, 25, 50, 75, 90, 95, 99]:
    print(f"   • {p}th percentile: {np.percentile(durations, p):.2f} seconds")

print(f"\n📦 DISTRIBUTION SHAPE:")
skewness = stats.skew(durations)
kurtosis = stats.kurtosis(durations)
print(f"   • Skewness: {skewness:.3f} {'(Right-skewed)' if skewness > 0 else '(Left-skewed)' if skewness < 0 else '(Symmetric)'}")
print(f"   • Kurtosis: {kurtosis:.3f} {'(Heavy-tailed)' if kurtosis > 0 else '(Light-tailed)' if kurtosis < 0 else '(Mesokurtic)'}")

print(f"\n🔍 OUTLIER ANALYSIS:")
print(f"   • Number of outliers: {len(outliers)} ({len(outliers)/len(durations)*100:.2f}%)")
print(f"   • Outlier range: < {lower_bound:.2f}s or > {upper_bound:.2f}s")
print(f"   • Outlier values: {', '.join([f'{x:.2f}' for x in outliers[:10]])}{'...' if len(outliers) > 10 else ''}")

print(f"\n⏱️  DURATION CATEGORIES:")
categories = {
    'Very Short (<5s)': len(durations[durations < 5]),
    'Short (5-10s)': len(durations[(durations >= 5) & (durations < 10)]),
    'Medium (10-20s)': len(durations[(durations >= 10) & (durations < 20)]),
    'Long (20-30s)': len(durations[(durations >= 20) & (durations < 30)]),
    'Very Long (>30s)': len(durations[durations >= 30])
}
for cat, count in categories.items():
    print(f"   • {cat}: {count:,} recordings ({count/len(durations)*100:.1f}%)")

print("\n" + "="*80)
print("✅ Analysis complete! High-resolution images saved:")
print("   • audio_duration_analysis_professional.png")
print("   • audio_duration_outlier_analysis.png")
print("="*80)