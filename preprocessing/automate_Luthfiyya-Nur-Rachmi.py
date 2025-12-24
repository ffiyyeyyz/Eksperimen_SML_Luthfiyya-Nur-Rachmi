import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# =========================
# Load Data
# =========================
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

# =========================
# Initial Cleaning
# =========================
def initial_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # drop kolom dengan missing value > 10%
    df = df.drop(columns=['company', 'agent'], errors='ignore')
    
    # hapus missing value
    df.dropna(inplace=True)
    
    # hapus duplikat
    df.drop_duplicates(inplace=True)
    
    # hapus data tamu = 0
    zero_guest = (
        (df['adults'] == 0) &
        (df['children'] == 0) &
        (df['babies'] == 0)
    )
    df = df[~zero_guest]
    
    return df



# =========================
# Konversi Tipe Data
# =========================
def conver_data_type(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'])
    
    return df

# =========================
# Feature Engineering
# =========================
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # binning lead_time
    df['lead_time_category'] = pd.cut(
        df['lead_time'],
        bins=[-1, 0.5, 1.5, 3, df['lead_time'].max()],
        labels=['Very Short', 'Short', 'Medium', 'Long']
    )
    
    return df

# =========================
# Outlier Handling (IQR)
# =========================
def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    
    return df

# =========================
# Encoding Categorical
# =========================
def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    le_country = LabelEncoder()
    df['country_encoded'] = le_country.fit_transform(df['country'])
    
    df.drop(columns=['country'], inplace=True)
    
    # one-hot encoding untuk selain country
    categorical_cols = df.select_dtypes(
        include=['object', 'category']
    ).columns.tolist()
    
    categorical_cols.remove('arrival_date_month')
    df = pd.get_dummies(
        df,
        columns=categorical_cols,
        drop_first=True
    )
    
    # mapping bulan
    month_mapping = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4,
        'May': 5, 'June': 6, 'July': 7,
        'August': 8, 'September': 9, 'October': 10,
        'November': 11, 'December': 12
    }
    
    df['arrival_date_month_num'] = df['arrival_date_month'].map(month_mapping)
    df.drop(columns=['arrival_date_month'], inplace=True)
        
    return df

# =========================
# Standardization
# =========================
def standardize_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    scaler = StandardScaler()
    
    numeric_cols = df.select_dtypes(
        include=['int64', 'float64']
    ).columns.tolist()
    
    numeric_cols.remove('is_canceled')
    
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df

# =========================
# Pipeline
# =========================
def preprocess_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    df = initial_cleaning(df)
    df = conver_data_type(df)
    df = feature_engineering(df)
    df = handle_outliers(df)
    df = encode_categorical_features(df)
    df = standardize_features(df)
    return df

# =========================
# Main
# =========================
if __name__ == "__main__":
    input_file = "../hotel_bookings_raw.csv"
    output_file = "hotel_bookings_preprocessing.csv"
    
    try:
        print("Loading data...")
        raw_df = load_data(input_file)
        
        print("Running preprocessing pipeline...")
        processed_df = preprocess_pipeline(raw_df)
        
        print(f"Saving processed data to {output_file}...")
        processed_df.to_csv(output_file, index=False)
        
        print("Done! Data saved successfully.")
        print(f"Final dataset shape: {processed_df.shape}")
        
    except FileNotFoundError:
        print(f"Error: File {input_file} tidak ditemukan.")
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")

