import pandas as pd
import numpy as np
import re
from collections import Counter
from textblob import TextBlob

def test_urgency_detection():
    """Test the urgency detection functionality"""
    print("🧪 Testing Urgency Detection System")
    print("=" * 50)
    
    # Load data
    print("Loading data...")
    df = pd.read_csv('parliamentary_data_clean.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"Loaded {len(df)} questions")
    
    # Test urgency keywords
    urgency_keywords = {
        'high': ['urgent', 'immediate', 'critical', 'emergency', 'crisis', 'disaster', 'outbreak', 'attack', 'breach', 'failure'],
        'medium': ['important', 'significant', 'serious', 'concerning', 'worrying', 'problem', 'issue', 'delay', 'shortage', 'risk'],
        'low': ['general', 'routine', 'regular', 'standard', 'normal', 'usual', 'typical', 'common']
    }
    
    def calculate_keyword_score(text):
        if pd.isna(text):
            return 0
        
        text_lower = text.lower()
        score = 0
        
        for keyword in urgency_keywords['high']:
            if keyword in text_lower:
                score += 3
        
        for keyword in urgency_keywords['medium']:
            if keyword in text_lower:
                score += 2
        
        for keyword in urgency_keywords['low']:
            if keyword in text_lower:
                score += 1
        
        return score
    
    # Calculate urgency scores
    print("Calculating urgency scores...")
    df['urgency_score'] = df['question'].apply(calculate_keyword_score)
    
    # Find high urgency questions
    high_urgency = df[df['urgency_score'] >= 3].copy()
    
    print(f"\n📊 URGENCY ANALYSIS RESULTS:")
    print(f"Total questions: {len(df)}")
    print(f"High urgency questions (score >= 3): {len(high_urgency)}")
    print(f"High urgency percentage: {(len(high_urgency) / len(df) * 100):.1f}%")
    
    # Show top urgency questions
    print(f"\n🏆 TOP 10 HIGHEST URGENCY QUESTIONS:")
    top_urgent = df.nlargest(10, 'urgency_score')[['date', 'department', 'deputy', 'heading', 'question', 'urgency_score']]
    
    for idx, row in top_urgent.iterrows():
        print(f"\n{idx+1}. Score: {row['urgency_score']}")
        print(f"   Date: {row['date'].strftime('%Y-%m-%d')}")
        print(f"   Department: {row['department']}")
        print(f"   Deputy: {row['deputy']}")
        print(f"   Heading: {row['heading']}")
        print(f"   Question: {row['question'][:150]}...")
        print("-" * 80)
    
    # Department analysis
    print(f"\n📋 DEPARTMENTS WITH HIGH URGENCY QUESTIONS:")
    dept_urgency = high_urgency['department'].value_counts().head(10)
    for dept, count in dept_urgency.items():
        print(f"   {dept}: {count} questions")
    
    # Deputy analysis
    print(f"\n👥 DEPUTIES ASKING HIGH URGENCY QUESTIONS:")
    deputy_urgency = high_urgency['deputy'].value_counts().head(10)
    for deputy, count in deputy_urgency.items():
        print(f"   {deputy}: {count} questions")
    
    # Keyword analysis
    print(f"\n🔍 COMMON KEYWORDS IN HIGH URGENCY QUESTIONS:")
    all_text = ' '.join(high_urgency['question'].fillna(''))
    words = re.findall(r'\b\w+\b', all_text.lower())
    word_freq = Counter(words)
    
    # Remove common stop words
    stop_words = set(['the', 'and', 'for', 'that', 'this', 'with', 'will', 'have', 'been', 'from', 'are', 'was', 'were', 'has', 'had', 'would', 'could', 'should'])
    filtered_words = {word: count for word, count in word_freq.items() 
                     if word not in stop_words and len(word) > 3}
    
    top_keywords = sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)[:15]
    for word, count in top_keywords:
        print(f"   {word}: {count}")
    
    print(f"\n✅ URGENCY DETECTION TEST COMPLETE!")

if __name__ == "__main__":
    test_urgency_detection() 