import pandas as pd
import json

def test_csv_loading():
    """Test CSV loading and identify any issues"""
    try:
        print("Testing CSV file loading...")
        
        # Try to load the CSV file
        df = pd.read_csv('Parliamentary Question dataset.csv')
        
        print(f"✅ Successfully loaded {len(df)} rows")
        print(f"Columns: {list(df.columns)}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"Unique departments: {df['department'].nunique()}")
        print(f"Unique deputies: {df['deputy'].nunique()}")
        
        # Check for any issues with the data
        print("\nChecking for data issues...")
        
        # Check for missing values
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            print("⚠️ Missing values found:")
            print(missing_data[missing_data > 0])
        else:
            print("✅ No missing values found")
        
        # Check for date parsing issues
        try:
            pd.to_datetime(df['date'])
            print("✅ Date parsing successful")
        except Exception as e:
            print(f"❌ Date parsing failed: {e}")
        
        # Check question length
        df['question_length'] = df['question'].str.len()
        print(f"✅ Question length stats:")
        print(f"   Min: {df['question_length'].min()}")
        print(f"   Max: {df['question_length'].max()}")
        print(f"   Mean: {df['question_length'].mean():.0f}")
        
        # Sample a few rows
        print("\nSample data (first 3 rows):")
        sample_data = df.head(3).to_dict('records')
        for i, row in enumerate(sample_data):
            print(f"\nRow {i+1}:")
            for key, value in row.items():
                if key == 'question':
                    print(f"  {key}: {str(value)[:100]}...")
                else:
                    print(f"  {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return False

if __name__ == "__main__":
    test_csv_loading() 