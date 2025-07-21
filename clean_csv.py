import pandas as pd
import re

def clean_csv():
    """Clean the CSV file to make it more compatible with D3.js"""
    print("Cleaning CSV file...")
    
    # Read the original CSV
    df = pd.read_csv('Parliamentary Question dataset.csv')
    
    # Clean the question column
    df['question'] = df['question'].astype(str).apply(lambda x: 
        re.sub(r'\s+', ' ', x).strip()  # Remove extra whitespace and line breaks
    )
    
    # Clean other text columns
    for col in ['department', 'heading', 'deputy']:
        df[col] = df[col].astype(str).str.strip()
    
    # Ensure date is in proper format
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    
    # Save cleaned CSV
    df.to_csv('parliamentary_data_clean.csv', index=False)
    
    print(f"✅ Cleaned CSV saved as 'parliamentary_data_clean.csv'")
    print(f"   Original rows: {len(df)}")
    print(f"   Sample cleaned question: {df['question'].iloc[0][:100]}...")
    
    return df

if __name__ == "__main__":
    clean_csv() 