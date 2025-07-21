import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# NLP Libraries
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

# ML Libraries
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb

# Visualization
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class UrgencyDetector:
    def __init__(self, data_path='parliamentary_data_clean.csv'):
        """Initialize the urgency detector"""
        self.data_path = data_path
        self.df = None
        self.urgency_scores = None
        self.models = {}
        self.urgency_keywords = {
            'high': ['urgent', 'immediate', 'critical', 'emergency', 'crisis', 'disaster', 'outbreak', 'attack', 'breach', 'failure'],
            'medium': ['important', 'significant', 'serious', 'concerning', 'worrying', 'problem', 'issue', 'delay', 'shortage', 'risk'],
            'low': ['general', 'routine', 'regular', 'standard', 'normal', 'usual', 'typical', 'common']
        }
        
    def load_data(self):
        """Load and prepare the data"""
        print("Loading parliamentary data...")
        self.df = pd.read_csv(self.data_path)
        
        # Convert date column
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        # Create basic features
        self.df['question_length'] = self.df['question'].str.len()
        self.df['word_count'] = self.df['question'].str.split().str.len()
        self.df['day_of_week'] = self.df['date'].dt.dayofweek
        self.df['month'] = self.df['date'].dt.month
        self.df['year'] = self.df['date'].dt.year
        
        print(f"Data loaded: {len(self.df)} questions")
        print(f"Date range: {self.df['date'].min()} to {self.df['date'].max()}")
        
        return self.df
    
    def create_urgency_features(self):
        """Create features for urgency detection"""
        print("\nCreating urgency features...")
        
        # 1. Keyword-based urgency scores
        self.df['urgency_keyword_score'] = self.df['question'].apply(self._calculate_keyword_score)
        
        # 2. Sentiment analysis
        self.df['sentiment_score'] = self.df['question'].apply(self._calculate_sentiment)
        
        # 3. Question type indicators
        self.df['has_urgent_indicators'] = self.df['question'].apply(self._has_urgent_indicators)
        self.df['has_time_indicators'] = self.df['question'].apply(self._has_time_indicators)
        self.df['has_action_indicators'] = self.df['question'].apply(self._has_action_indicators)
        
        # 4. Department urgency weights
        self.df['department_urgency_weight'] = self.df['department'].apply(self._get_department_urgency_weight)
        
        # 5. Deputy activity level
        deputy_activity = self.df['deputy'].value_counts()
        self.df['deputy_activity_level'] = self.df['deputy'].map(deputy_activity)
        
        # 6. Question complexity
        self.df['question_complexity'] = self.df['question'].apply(self._calculate_complexity)
        
        # 7. Temporal urgency (recent questions might be more urgent)
        max_date = self.df['date'].max()
        self.df['days_since_max'] = (max_date - self.df['date']).dt.days
        self.df['temporal_urgency'] = 1 / (1 + self.df['days_since_max'])
        
        print(f"Features created. Shape: {self.df.shape}")
        return self.df
    
    def _calculate_keyword_score(self, text):
        """Calculate urgency score based on keywords"""
        if pd.isna(text):
            return 0
        
        text_lower = text.lower()
        score = 0
        
        # High urgency keywords
        for keyword in self.urgency_keywords['high']:
            if keyword in text_lower:
                score += 3
        
        # Medium urgency keywords
        for keyword in self.urgency_keywords['medium']:
            if keyword in text_lower:
                score += 2
        
        # Low urgency keywords
        for keyword in self.urgency_keywords['low']:
            if keyword in text_lower:
                score += 1
        
        return score
    
    def _calculate_sentiment(self, text):
        """Calculate sentiment score"""
        if pd.isna(text):
            return 0
        
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity
        except:
            return 0
    
    def _has_urgent_indicators(self, text):
        """Check for urgent indicators in text"""
        if pd.isna(text):
            return 0
        
        urgent_patterns = [
            r'\b(urgent|immediate|critical|emergency|crisis)\b',
            r'\b(now|today|asap|immediately)\b',
            r'\b(breakdown|failure|outage|disaster)\b',
            r'\b(attack|breach|security|threat)\b'
        ]
        
        text_lower = text.lower()
        for pattern in urgent_patterns:
            if re.search(pattern, text_lower):
                return 1
        return 0
    
    def _has_time_indicators(self, text):
        """Check for time-related urgency indicators"""
        if pd.isna(text):
            return 0
        
        time_patterns = [
            r'\b(deadline|due date|timeline)\b',
            r'\b(delay|postponed|overdue)\b',
            r'\b(schedule|appointment|meeting)\b',
            r'\b(time|period|duration)\b'
        ]
        
        text_lower = text.lower()
        for pattern in time_patterns:
            if re.search(pattern, text_lower):
                return 1
        return 0
    
    def _has_action_indicators(self, text):
        """Check for action-required indicators"""
        if pd.isna(text):
            return 0
        
        action_patterns = [
            r'\b(action|response|reply|answer)\b',
            r'\b(investigation|review|examination)\b',
            r'\b(decision|resolution|solution)\b',
            r'\b(implement|execute|carry out)\b'
        ]
        
        text_lower = text.lower()
        for pattern in action_patterns:
            if re.search(pattern, text_lower):
                return 1
        return 0
    
    def _get_department_urgency_weight(self, department):
        """Assign urgency weights to departments"""
        high_urgency_depts = ['Health', 'Justice', 'Defence', 'Finance', 'Transport']
        medium_urgency_depts = ['Education', 'Housing', 'Environment', 'Social Protection']
        
        if department in high_urgency_depts:
            return 3
        elif department in medium_urgency_depts:
            return 2
        else:
            return 1
    
    def _calculate_complexity(self, text):
        """Calculate question complexity score"""
        if pd.isna(text):
            return 0
        
        # Simple complexity metrics
        sentences = len(re.split(r'[.!?]+', text))
        words = len(text.split())
        avg_sentence_length = words / max(sentences, 1)
        
        # Complexity score based on length and structure
        complexity = (avg_sentence_length * 0.3) + (sentences * 0.2) + (words * 0.01)
        return min(complexity, 10)  # Cap at 10
    
    def create_urgency_labels(self, method='composite'):
        """Create urgency labels for training"""
        print(f"\nCreating urgency labels using {method} method...")
        
        if method == 'composite':
            # Create composite urgency score
            features = [
                'urgency_keyword_score',
                'sentiment_score',
                'has_urgent_indicators',
                'has_time_indicators',
                'has_action_indicators',
                'department_urgency_weight',
                'question_complexity',
                'temporal_urgency'
            ]
            
            # Normalize features
            scaler = StandardScaler()
            urgency_features = self.df[features].fillna(0)
            urgency_features_scaled = scaler.fit_transform(urgency_features)
            
            # Create composite score
            weights = [0.25, 0.15, 0.20, 0.10, 0.10, 0.10, 0.05, 0.05]
            composite_score = np.average(urgency_features_scaled, axis=1, weights=weights)
            
            # Create urgency labels
            self.df['urgency_score'] = composite_score
            self.df['urgency_level'] = pd.cut(composite_score, 
                                            bins=3, 
                                            labels=['Low', 'Medium', 'High'])
            
        elif method == 'keyword_based':
            # Simple keyword-based labeling
            self.df['urgency_level'] = pd.cut(self.df['urgency_keyword_score'], 
                                            bins=3, 
                                            labels=['Low', 'Medium', 'High'])
        
        # Display urgency distribution
        urgency_dist = self.df['urgency_level'].value_counts()
        print("\nUrgency Level Distribution:")
        print(urgency_dist)
        
        return self.df
    
    def train_urgency_models(self):
        """Train ML models for urgency classification"""
        print("\n=== TRAINING URGENCY CLASSIFICATION MODELS ===")
        
        # Prepare features
        feature_columns = [
            'urgency_keyword_score', 'sentiment_score', 'has_urgent_indicators',
            'has_time_indicators', 'has_action_indicators', 'department_urgency_weight',
            'question_complexity', 'temporal_urgency', 'question_length', 'word_count',
            'day_of_week', 'month', 'deputy_activity_level'
        ]
        
        X = self.df[feature_columns].fillna(0)
        y = self.df['urgency_level']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Define models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'predictions': y_pred,
                'classification_report': classification_report(y_test, y_pred)
            }
            
            print(f"{name} - Accuracy: {accuracy:.3f}")
            print(f"Classification Report:\n{results[name]['classification_report']}")
        
        self.models = results
        return results
    
    def analyze_high_urgency_questions(self):
        """Analyze high urgency questions"""
        print("\n=== HIGH URGENCY QUESTIONS ANALYSIS ===")
        
        high_urgency = self.df[self.df['urgency_level'] == 'High'].copy()
        
        print(f"Total high urgency questions: {len(high_urgency)}")
        print(f"Percentage of total: {(len(high_urgency) / len(self.df) * 100):.1f}%")
        
        # Top departments with high urgency questions
        print("\nTop Departments with High Urgency Questions:")
        dept_urgency = high_urgency['department'].value_counts().head(10)
        print(dept_urgency)
        
        # Top deputies asking high urgency questions
        print("\nTop Deputies Asking High Urgency Questions:")
        deputy_urgency = high_urgency['deputy'].value_counts().head(10)
        print(deputy_urgency)
        
        # Common keywords in high urgency questions
        print("\nCommon Keywords in High Urgency Questions:")
        all_text = ' '.join(high_urgency['question'].fillna(''))
        words = re.findall(r'\b\w+\b', all_text.lower())
        word_freq = Counter(words)
        
        # Remove common stop words
        stop_words = set(['the', 'and', 'for', 'that', 'this', 'with', 'will', 'have', 'been', 'from'])
        filtered_words = {word: count for word, count in word_freq.items() 
                         if word not in stop_words and len(word) > 3}
        
        top_keywords = sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)[:15]
        for word, count in top_keywords:
            print(f"  {word}: {count}")
        
        return high_urgency
    
    def create_urgency_visualizations(self):
        """Create visualizations for urgency analysis"""
        print("\n=== CREATING URGENCY VISUALIZATIONS ===")
        
        # 1. Urgency distribution
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Urgency level distribution
        urgency_counts = self.df['urgency_level'].value_counts()
        axes[0, 0].pie(urgency_counts.values, labels=urgency_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('Urgency Level Distribution')
        
        # Department urgency heatmap
        dept_urgency = self.df.groupby(['department', 'urgency_level']).size().unstack(fill_value=0)
        top_depts = dept_urgency.sum(axis=1).nlargest(10).index
        dept_urgency_top = dept_urgency.loc[top_depts]
        
        sns.heatmap(dept_urgency_top, annot=True, fmt='d', cmap='YlOrRd', ax=axes[0, 1])
        axes[0, 1].set_title('Department vs Urgency Level')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Urgency score distribution
        axes[1, 0].hist(self.df['urgency_score'], bins=30, alpha=0.7, color='skyblue')
        axes[1, 0].set_title('Urgency Score Distribution')
        axes[1, 0].set_xlabel('Urgency Score')
        axes[1, 0].set_ylabel('Frequency')
        
        # Urgency over time
        daily_urgency = self.df.groupby('date')['urgency_score'].mean()
        axes[1, 1].plot(daily_urgency.index, daily_urgency.values, linewidth=2)
        axes[1, 1].set_title('Average Urgency Score Over Time')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Average Urgency Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('urgency_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Interactive plotly visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Urgency by Department', 'Urgency Score Distribution', 
                          'Top High Urgency Keywords', 'Urgency Over Time'),
            specs=[[{"type": "bar"}, {"type": "histogram"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Department urgency
        dept_avg_urgency = self.df.groupby('department')['urgency_score'].mean().sort_values(ascending=True)
        fig.add_trace(
            go.Bar(x=dept_avg_urgency.values, y=dept_avg_urgency.index, orientation='h', name='Department'),
            row=1, col=1
        )
        
        # Urgency score histogram
        fig.add_trace(
            go.Histogram(x=self.df['urgency_score'], nbinsx=30, name='Score Distribution'),
            row=1, col=2
        )
        
        # Top keywords (simplified)
        high_urgency = self.df[self.df['urgency_level'] == 'High']
        if len(high_urgency) > 0:
            all_text = ' '.join(high_urgency['question'].fillna(''))
            words = re.findall(r'\b\w+\b', all_text.lower())
            word_freq = Counter(words)
            stop_words = set(['the', 'and', 'for', 'that', 'this', 'with', 'will', 'have', 'been', 'from'])
            filtered_words = {word: count for word, count in word_freq.items() 
                             if word not in stop_words and len(word) > 3}
            top_keywords = sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)[:10]
            
            if top_keywords:
                keywords, counts = zip(*top_keywords)
                fig.add_trace(
                    go.Bar(x=keywords, y=counts, name='Keywords'),
                    row=2, col=1
                )
        
        # Urgency over time
        daily_urgency = self.df.groupby('date')['urgency_score'].mean()
        fig.add_trace(
            go.Scatter(x=daily_urgency.index, y=daily_urgency.values, mode='lines', name='Daily Urgency'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="Parliamentary Questions Urgency Analysis")
        fig.write_html('urgency_analysis_interactive.html')
        
        print("Visualizations saved as 'urgency_analysis.png' and 'urgency_analysis_interactive.html'")
    
    def get_high_priority_questions(self, top_n=20):
        """Get the top N highest priority questions"""
        print(f"\n=== TOP {top_n} HIGHEST PRIORITY QUESTIONS ===")
        
        # Sort by urgency score
        high_priority = self.df.nlargest(top_n, 'urgency_score')[['date', 'department', 'deputy', 'heading', 'question', 'urgency_score', 'urgency_level']]
        
        print(f"\nTop {top_n} Highest Priority Questions:")
        for idx, row in high_priority.iterrows():
            print(f"\n{idx+1}. Score: {row['urgency_score']:.2f} | Level: {row['urgency_level']}")
            print(f"   Date: {row['date'].strftime('%Y-%m-%d')}")
            print(f"   Department: {row['department']}")
            print(f"   Deputy: {row['deputy']}")
            print(f"   Heading: {row['heading']}")
            print(f"   Question: {row['question'][:200]}...")
            print("-" * 80)
        
        return high_priority
    
    def run_complete_analysis(self):
        """Run the complete urgency analysis"""
        print("🚀 STARTING URGENCY DETECTION ANALYSIS")
        print("=" * 60)
        
        # 1. Load data
        self.load_data()
        
        # 2. Create features
        self.create_urgency_features()
        
        # 3. Create urgency labels
        self.create_urgency_labels()
        
        # 4. Train models
        self.train_urgency_models()
        
        # 5. Analyze high urgency questions
        high_urgency_analysis = self.analyze_high_urgency_questions()
        
        # 6. Create visualizations
        self.create_urgency_visualizations()
        
        # 7. Get top priority questions
        top_priority = self.get_high_priority_questions()
        
        print("\n✅ URGENCY ANALYSIS COMPLETE!")
        print("=" * 60)
        
        return {
            'high_urgency_analysis': high_urgency_analysis,
            'top_priority_questions': top_priority,
            'models': self.models,
            'urgency_distribution': self.df['urgency_level'].value_counts()
        }

if __name__ == "__main__":
    # Initialize and run the analysis
    detector = UrgencyDetector()
    results = detector.run_complete_analysis()
    
    # Print summary
    print("\n📊 URGENCY ANALYSIS SUMMARY:")
    print(f"Total questions analyzed: {len(detector.df)}")
    print(f"High urgency questions: {len(results['high_urgency_analysis'])}")
    print(f"High urgency percentage: {(len(results['high_urgency_analysis']) / len(detector.df) * 100):.1f}%")
    
    best_model = max(results['models'].items(), key=lambda x: x[1]['accuracy'])
    print(f"Best model: {best_model[0]} (Accuracy: {best_model[1]['accuracy']:.3f})")
    
    print(f"\nUrgency Distribution:")
    print(results['urgency_distribution']) 