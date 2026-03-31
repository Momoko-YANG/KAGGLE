# Define Paths
BASE_PATH = Path('/kaggle/input/competitions/birdclef-2026')
TRAIN_AUDIO_PATH = BASE_PATH / 'train_audio'
TEST_SOUNDSCAPES_PATH = BASE_PATH / 'test_soundscapes'
TRAIN_SOUNDSCAPES_PATH = BASE_PATH / 'train_soundscapes'

# Load all data files
print("Loading data files...")
train_df = pd.read_csv(BASE_PATH / 'train.csv')
taxonomy_df = pd.read_csv(BASE_PATH / 'taxonomy.csv')
sample_submission = pd.read_csv(BASE_PATH / 'sample_submission.csv')
train_soundscapes_labels = pd.read_csv(BASE_PATH / 'train_soundscapes_labels.csv')
recording_location = pd.read_csv(BASE_PATH / 'recording_location.txt', sep='\t')

# Display dataset overview
print("\n" + "="*60)
print("DATASET OVERVIEW")
print("="*60)
print(f"Training samples: {len(train_df):, }")
print(f"Unique bird species: {train_df['primary_label'].nunique():,}")
print(f"Train soundscapes: {len(train_soundscapes_labels):,}")
print(f"Test soundscapes: {len(sample_submission):,}")
print(f"Taxonomy entries: {len(taxonomy_df):,}")
print(f"Recording locations: {len(recording_location):,}")

# Display first few rows of each dataset
print("\n" + "="*60)
print("TRAIN DATA SAMPLE")
print("="*60)
display(train_df.head())

print("\n" + "="*60)
print("TAXONOMY DATA SAMPLE")
print("="*60)
display(train_soundscapes_labels.head())

# Check for missing values
print("\n" + "="*60)
print("MISSING VALUES CHECK")
print("="*60)
print("Train data missing values:\n", train_df.isnull().sum())
print("\nTaxonomy missing values:\n", taxonomy_df.isnull().sum())
