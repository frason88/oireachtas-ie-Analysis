import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

def load_data():
    """Load the parliamentary questions dataset"""
    df = pd.read_csv('Parliamentary Question dataset.csv')
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['weekday'] = df['date'].dt.day_name()
    return df

def analyze_headings(df):
    """Analyze question headings/categories"""
    print("\n" + "="*60)
    print("QUESTION HEADINGS ANALYSIS")
    print("="*60)
    
    heading_counts = df['heading'].value_counts()
    print(f"Total unique headings: {len(heading_counts)}")
    print("\nTop 10 Question Headings:")
    for i, (heading, count) in enumerate(heading_counts.head(10).items(), 1):
        percentage = (count / len(df)) * 100
        print(f"{i:2d}. {heading}: {count:,} questions ({percentage:.1f}%)")

def analyze_question_patterns(df):
    """Analyze patterns in question text"""
    print("\n" + "="*60)
    print("QUESTION TEXT ANALYSIS")
    print("="*60)
    
    # Question length analysis
    df['question_length'] = df['question'].str.len()
    df['word_count'] = df['question'].str.split().str.len()
    
    print(f"Question Length Statistics:")
    print(f"  Average length: {df['question_length'].mean():.0f} characters")
    print(f"  Median length: {df['question_length'].median():.0f} characters")
    print(f"  Min length: {df['question_length'].min():.0f} characters")
    print(f"  Max length: {df['question_length'].max():.0f} characters")
    
    print(f"\nWord Count Statistics:")
    print(f"  Average words: {df['word_count'].mean():.1f} words")
    print(f"  Median words: {df['word_count'].median():.1f} words")
    
    # Find common words
    all_words = []
    for question in df['question'].dropna():
        words = re.findall(r'\b[a-zA-Z]{3,}\b', question.lower())
        all_words.extend(words)
    
    word_counts = Counter(all_words)
    print(f"\nMost Common Words (excluding common stop words):")
    stop_words = {'the', 'and', 'for', 'that', 'this', 'with', 'will', 'have', 'been', 'from', 'they', 'their', 'said', 'each', 'which', 'she', 'will', 'would', 'there', 'could', 'been', 'call', 'first', 'who', 'its', 'now', 'her', 'has', 'more', 'when', 'an', 'had', 'by', 'word', 'but', 'not', 'what', 'all', 'were', 'we', 'can', 'out', 'other', 'about', 'many', 'then', 'them', 'these', 'so', 'some', 'her', 'would', 'make', 'like', 'into', 'him', 'time', 'two', 'more', 'go', 'no', 'way', 'could', 'my', 'than', 'first', 'been', 'call', 'who', 'its', 'now', 'find', 'long', 'down', 'day', 'did', 'get', 'come', 'made', 'may', 'part'}
    
    filtered_words = {word: count for word, count in word_counts.items() 
                     if word not in stop_words and len(word) > 3}
    
    for i, (word, count) in enumerate(sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)[:15], 1):
        print(f"{i:2d}. {word}: {count:,} occurrences")

def analyze_temporal_patterns(df):
    """Analyze temporal patterns in the data"""
    print("\n" + "="*60)
    print("TEMPORAL PATTERNS ANALYSIS")
    print("="*60)
    
    # Questions by weekday
    weekday_counts = df['weekday'].value_counts().reindex([
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
    ])
    
    print("Questions by Day of Week:")
    for day, count in weekday_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {day}: {count:,} questions ({percentage:.1f}%)")
    
    # Questions by month
    month_counts = df['month'].value_counts().sort_index()
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    print(f"\nQuestions by Month:")
    for month_num, count in month_counts.items():
        month_name = month_names[month_num - 1]
        percentage = (count / len(df)) * 100
        print(f"  {month_name}: {count:,} questions ({percentage:.1f}%)")

def analyze_department_deputy_relationships(df):
    """Analyze relationships between departments and deputies"""
    print("\n" + "="*60)
    print("DEPARTMENT-DEPUTY RELATIONSHIPS")
    print("="*60)
    
    # Top deputies by department
    dept_deputy_counts = df.groupby(['department', 'deputy']).size().reset_index(name='count')
    
    print("Top 3 Deputies by Department:")
    for dept in df['department'].value_counts().head(5).index:
        dept_data = dept_deputy_counts[dept_deputy_counts['department'] == dept]
        top_deputies = dept_data.nlargest(3, 'count')
        print(f"\n{dept}:")
        for _, row in top_deputies.iterrows():
            print(f"  - {row['deputy']}: {row['count']:,} questions")

def create_additional_visualizations(df):
    """Create additional visualizations"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Additional Parliamentary Questions Analysis', fontsize=16, fontweight='bold')
    
    # 1. Questions by weekday
    weekday_counts = df['weekday'].value_counts().reindex([
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
    ])
    axes[0, 0].bar(weekday_counts.index, weekday_counts.values, color='lightcoral')
    axes[0, 0].set_title('Questions by Day of Week')
    axes[0, 0].set_ylabel('Number of Questions')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Question length vs department
    dept_length = df.groupby('department')['question_length'].mean().sort_values(ascending=False).head(10)
    axes[0, 1].barh(dept_length.index, dept_length.values, color='lightblue')
    axes[0, 1].set_title('Average Question Length by Department')
    axes[0, 1].set_xlabel('Average Characters')
    
    # 3. Top headings
    heading_counts = df['heading'].value_counts().head(10)
    axes[1, 0].barh(heading_counts.index, heading_counts.values, color='lightgreen')
    axes[1, 0].set_title('Top 10 Question Headings')
    axes[1, 0].set_xlabel('Number of Questions')
    
    # 4. Questions per deputy distribution
    deputy_counts = df['deputy'].value_counts()
    axes[1, 1].hist(deputy_counts.values, bins=30, color='purple', alpha=0.7)
    axes[1, 1].set_title('Distribution of Questions per Deputy')
    axes[1, 1].set_xlabel('Number of Questions')
    axes[1, 1].set_ylabel('Number of Deputies')
    
    plt.tight_layout()
    plt.savefig('detailed_analysis.png', dpi=300, bbox_inches='tight')
    print("\nDetailed analysis visualizations saved as 'detailed_analysis.png'")
    plt.show()

if __name__ == "__main__":
    print("Loading data for detailed analysis...")
    df = load_data()
    
    analyze_headings(df)
    analyze_question_patterns(df)
    analyze_temporal_patterns(df)
    analyze_department_deputy_relationships(df)
    
    print("\nCreating additional visualizations...")
    create_additional_visualizations(df)
    
    print("\nDetailed analysis complete!") 