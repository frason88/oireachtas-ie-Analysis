import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_clean_data():
    """Load and clean the parliamentary questions dataset"""
    print("Loading dataset...")
    df = pd.read_csv('Parliamentary Question dataset.csv')
    
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Extract year and month for analysis
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['month_name'] = df['date'].dt.strftime('%B')
    
    print(f"Dataset loaded: {len(df)} questions from {df['date'].min()} to {df['date'].max()}")
    return df

def create_dashboard():
    """Create comprehensive dashboard with multiple visualizations"""
    df = load_and_clean_data()
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Questions by Department (Top 10)',
            'Questions Over Time',
            'Questions by Deputy (Top 15)',
            'Questions by Month',
            'Department Activity Heatmap',
            'Question Length Distribution'
        ),
        specs=[[{"type": "bar"}, {"type": "scatter"}],
               [{"type": "bar"}, {"type": "bar"}],
               [{"type": "heatmap"}, {"type": "histogram"}]]
    )
    
    # 1. Questions by Department (Top 10)
    dept_counts = df['department'].value_counts().head(10)
    fig.add_trace(
        go.Bar(x=dept_counts.values, y=dept_counts.index, orientation='h', 
               name='Department', marker_color='lightblue'),
        row=1, col=1
    )
    
    # 2. Questions Over Time
    time_series = df.groupby('date').size().reset_index(name='count')
    fig.add_trace(
        go.Scatter(x=time_series['date'], y=time_series['count'], 
                  mode='lines', name='Daily Questions', line=dict(color='red')),
        row=1, col=2
    )
    
    # 3. Questions by Deputy (Top 15)
    deputy_counts = df['deputy'].value_counts().head(15)
    fig.add_trace(
        go.Bar(x=deputy_counts.index, y=deputy_counts.values, 
               name='Deputy', marker_color='lightgreen'),
        row=2, col=1
    )
    
    # 4. Questions by Month
    month_counts = df['month_name'].value_counts().reindex([
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ])
    fig.add_trace(
        go.Bar(x=month_counts.index, y=month_counts.values, 
               name='Month', marker_color='orange'),
        row=2, col=2
    )
    
    # 5. Department Activity Heatmap
    dept_month = df.groupby(['department', 'month_name']).size().unstack(fill_value=0)
    top_depts = df['department'].value_counts().head(8).index
    dept_month_filtered = dept_month.loc[top_depts]
    
    fig.add_trace(
        go.Heatmap(z=dept_month_filtered.values, 
                  x=dept_month_filtered.columns,
                  y=dept_month_filtered.index,
                  colorscale='Viridis', name='Activity'),
        row=3, col=1
    )
    
    # 6. Question Length Distribution
    df['question_length'] = df['question'].str.len()
    fig.add_trace(
        go.Histogram(x=df['question_length'], nbinsx=50, 
                    name='Length', marker_color='purple'),
        row=3, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=1200, width=1400,
        title_text="Irish Parliamentary Questions Analysis Dashboard",
        showlegend=False
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Number of Questions", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=2)
    fig.update_xaxes(title_text="Deputy", row=2, col=1)
    fig.update_xaxes(title_text="Month", row=2, col=2)
    fig.update_xaxes(title_text="Month", row=3, col=1)
    fig.update_xaxes(title_text="Question Length (characters)", row=3, col=2)
    
    fig.update_yaxes(title_text="Department", row=1, col=1)
    fig.update_yaxes(title_text="Number of Questions", row=1, col=2)
    fig.update_yaxes(title_text="Number of Questions", row=2, col=1)
    fig.update_yaxes(title_text="Number of Questions", row=2, col=2)
    fig.update_yaxes(title_text="Department", row=3, col=1)
    fig.update_yaxes(title_text="Frequency", row=3, col=2)
    
    fig.write_html('parliamentary_dashboard.html')
    print("Dashboard saved as 'parliamentary_dashboard.html'")
    
    return fig

def create_matplotlib_visualizations(df):
    """Create static matplotlib visualizations"""
    
    # Set up the figure
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Irish Parliamentary Questions Analysis', fontsize=16, fontweight='bold')
    
    # 1. Top Departments
    dept_counts = df['department'].value_counts().head(10)
    axes[0, 0].barh(dept_counts.index, dept_counts.values, color='skyblue')
    axes[0, 0].set_title('Top 10 Departments by Questions')
    axes[0, 0].set_xlabel('Number of Questions')
    
    # 2. Questions Over Time (Monthly)
    monthly_counts = df.groupby([df['date'].dt.to_period('M')]).size()
    axes[0, 1].plot(range(len(monthly_counts)), monthly_counts.values, linewidth=2, color='red')
    axes[0, 1].set_title('Questions Over Time (Monthly)')
    axes[0, 1].set_xlabel('Month')
    axes[0, 1].set_ylabel('Number of Questions')
    
    # 3. Top Deputies
    deputy_counts = df['deputy'].value_counts().head(15)
    axes[0, 2].bar(range(len(deputy_counts)), deputy_counts.values, color='lightgreen')
    axes[0, 2].set_title('Top 15 Deputies by Questions')
    axes[0, 2].set_xlabel('Deputy')
    axes[0, 2].set_ylabel('Number of Questions')
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # 4. Questions by Month
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    month_counts = df['month_name'].value_counts().reindex(month_order)
    axes[1, 0].bar(month_counts.index, month_counts.values, color='orange')
    axes[1, 0].set_title('Questions by Month')
    axes[1, 0].set_xlabel('Month')
    axes[1, 0].set_ylabel('Number of Questions')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 5. Question Length Distribution
    df['question_length'] = df['question'].str.len()
    axes[1, 1].hist(df['question_length'], bins=50, color='purple', alpha=0.7)
    axes[1, 1].set_title('Question Length Distribution')
    axes[1, 1].set_xlabel('Question Length (characters)')
    axes[1, 1].set_ylabel('Frequency')
    
    # 6. Department vs Year Heatmap
    dept_year = df.groupby(['department', 'year']).size().unstack(fill_value=0)
    top_depts = df['department'].value_counts().head(8).index
    dept_year_filtered = dept_year.loc[top_depts]
    
    im = axes[1, 2].imshow(dept_year_filtered.values, cmap='YlOrRd', aspect='auto')
    axes[1, 2].set_title('Department Activity by Year')
    axes[1, 2].set_xlabel('Year')
    axes[1, 2].set_ylabel('Department')
    axes[1, 2].set_xticks(range(len(dept_year_filtered.columns)))
    axes[1, 2].set_xticklabels(dept_year_filtered.columns)
    axes[1, 2].set_yticks(range(len(dept_year_filtered.index)))
    axes[1, 2].set_yticklabels(dept_year_filtered.index)
    
    # Add colorbar
    plt.colorbar(im, ax=axes[1, 2])
    
    plt.tight_layout()
    plt.savefig('parliamentary_analysis.png', dpi=300, bbox_inches='tight')
    print("Static visualizations saved as 'parliamentary_analysis.png'")
    plt.show()

def generate_summary_statistics(df):
    """Generate and display summary statistics"""
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    
    print(f"Total Questions: {len(df):,}")
    print(f"Date Range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
    print(f"Unique Departments: {df['department'].nunique()}")
    print(f"Unique Deputies: {df['deputy'].nunique()}")
    print(f"Average Question Length: {df['question'].str.len().mean():.0f} characters")
    
    print(f"\nTop 5 Departments:")
    for i, (dept, count) in enumerate(df['department'].value_counts().head().items(), 1):
        print(f"{i}. {dept}: {count:,} questions")
    
    print(f"\nTop 5 Most Active Deputies:")
    for i, (deputy, count) in enumerate(df['deputy'].value_counts().head().items(), 1):
        print(f"{i}. {deputy}: {count:,} questions")
    
    print(f"\nQuestions by Year:")
    year_counts = df['year'].value_counts().sort_index()
    for year, count in year_counts.items():
        print(f"{year}: {count:,} questions")

if __name__ == "__main__":
    # Load data
    df = load_and_clean_data()
    
    # Generate summary statistics
    generate_summary_statistics(df)
    
    # Create interactive dashboard
    print("\nCreating interactive dashboard...")
    create_dashboard()
    
    # Create static visualizations
    print("\nCreating static visualizations...")
    create_matplotlib_visualizations(df)
    
    print("\nAnalysis complete! Check the generated files:")
    print("- parliamentary_dashboard.html (interactive dashboard)")
    print("- parliamentary_analysis.png (static visualizations)") 