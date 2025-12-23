import pandas as pd
import numpy as np

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def initial_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    cols_to_drop = ['company', 'agent']
    df = df.drop(columns=cols_to_drop, errors='ignore')
    df.dropna(inplace=True)
    
    filter_zero_guests = (df['children'] == 0) & (df['babies'] == 0) & (df['adults'] == 0)
    df = df[~filter_zero_guests]
    
    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    def categorize_lead_time(lead_time):
        if lead_time <= 7:
            return 'Last Minute'
        elif lead_time <= 30:
            return 'Short Term'
        elif lead_time <= 90:
            return 'Medium Term'
        else:
            return 'Long Term'
    
    df['lead_time_category'] = df['lead_time'].apply(categorize_lead_time)
    
    return df

def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    categorical_cols = [
        'hotel',
        'meal',
        'market_segment',
        'customer_type',
        'lead_time_category' 
    ]
    
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    return df

def preprocess_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    df = initial_cleaning(df)
    df = feature_engineering(df)
    df = encode_categorical_features(df)
    return df

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
        print(f"Shape of processed data: {processed_df.shape}")
        
    except FileNotFoundError:
        print(f"Error: File {input_file} tidak ditemukan.")
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")
