# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set page config
st.set_page_config(page_title="Pharma Industry Analysis", layout="wide")

class DataAnalyzer:
    def __init__(self, df):
        self.df = df
        self.prepare_data()
    
    def prepare_data(self):
        # Create binary columns for Yes/No values
        binary_columns = ['Manufacturer', 'Brand', 'Distributor', 'F&B (Food & Beverage)',
                         'Probiotics', 'Fortification']
        for col in binary_columns:
            self.df[f'{col}_Binary'] = (self.df[col] == 'Yes').astype(int)
        
        # Split multiple categories into lists
        list_columns = ['Product Categories', 'Health Segments', 'Certifications', 
                       'Target Audience', 'Innovation/Research Areas']
        for col in list_columns:
            if col in self.df.columns:
                self.df[f'{col}_List'] = self.df[col].str.split(',')
    
    def calculate_company_scores(self):
        scores = pd.DataFrame()
        scores['Company'] = self.df['Company Name']
        
        # Calculate various scores
        if 'Certifications' in self.df.columns:
            scores['Certification_Score'] = self.df['Certifications'].str.count(',') + 1
        
        scores['Market_Presence'] = self.df['Geographical Presence'].apply(
            lambda x: 10 if 'Global' in x else 5)
        
        if 'Innovation/Research Areas' in self.df.columns:
            scores['Innovation_Score'] = self.df['Innovation/Research Areas'].str.count(',') + 1
        
        scores['Product_Diversity'] = self.df['Product Categories'].str.count(',') + 1
        
        scores['Total_Score'] = scores.select_dtypes(include=[np.number]).sum(axis=1)
        return scores
    
    def analyze_market_segments(self):
        segments = {}
        if 'Health Segments' in self.df.columns:
            for _, row in self.df.iterrows():
                if pd.notna(row['Health Segments']):
                    for segment in row['Health Segments'].split(','):
                        segment = segment.strip()
                        segments[segment] = segments.get(segment, 0) + 1
        return pd.DataFrame(list(segments.items()), columns=['Segment', 'Count'])

def create_matplotlib_figure(figure_func):
    fig, ax = plt.subplots(figsize=(10, 6))
    figure_func(ax)
    plt.tight_layout()
    return fig

def create_visualizations(analyzer):
    st.title("Pharmaceutical Industry Analysis Dashboard")
    
    # Sidebar filters
    st.sidebar.header("Filters")
    company_type = st.sidebar.multiselect(
        "Select Company Type",
        options=analyzer.df['Company Type'].unique(),
        default=analyzer.df['Company Type'].unique()
    )
    
    # Filter data
    filtered_df = analyzer.df[analyzer.df['Company Type'].isin(company_type)]
    
    # Company Type Distribution
    st.subheader("Company Type Distribution")
    fig = create_matplotlib_figure(lambda ax: 
        sns.countplot(data=filtered_df, x='Company Type', ax=ax)
    )
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # Manufacturing Capabilities
    st.subheader("Manufacturing Capabilities")
    manufacturing_cols = ['Manufacturer_Binary', 'Brand_Binary', 'Distributor_Binary']
    manufacturing_data = filtered_df[manufacturing_cols].sum()
    fig = create_matplotlib_figure(lambda ax: 
        manufacturing_data.plot(kind='bar', ax=ax)
    )
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # Geographic Presence
    st.subheader("Geographic Presence")
    presence_data = filtered_df['Geographical Presence'].value_counts()
    fig = create_matplotlib_figure(lambda ax: 
        presence_data.plot(kind='barh', ax=ax)
    )
    st.pyplot(fig)
    
    # Health Segments Analysis
    st.subheader("Health Segments Distribution")
    segments = analyzer.analyze_market_segments()
    if not segments.empty:
        fig = create_matplotlib_figure(lambda ax: 
            sns.barplot(data=segments, x='Count', y='Segment', ax=ax)
        )
        st.pyplot(fig)
    
    # Company Performance Metrics
    st.subheader("Company Performance Analysis")
    scores = analyzer.calculate_company_scores()
    fig = create_matplotlib_figure(lambda ax: 
        sns.scatterplot(data=scores, x='Market_Presence', y='Innovation_Score', 
                       size='Product_Diversity', ax=ax)
    )
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # Top Companies Analysis
    st.subheader("Top Companies by Total Score")
    top_companies = scores.nlargest(10, 'Total_Score')
    fig = create_matplotlib_figure(lambda ax: 
        sns.barplot(data=top_companies, x='Total_Score', y='Company', ax=ax)
    )
    st.pyplot(fig)
    
    # Product Categories Analysis
    st.subheader("Product Categories Distribution")
    product_cats = filtered_df['Product Categories'].str.split(',').explode().value_counts()
    fig = create_matplotlib_figure(lambda ax: 
        product_cats.head(10).plot(kind='barh', ax=ax)
    )
    st.pyplot(fig)
    
    # Detailed Analysis Tables
    st.header("Detailed Analysis")
    
    # Company Scores Table
    st.subheader("Company Scores")
    st.dataframe(scores)
    
    # Summary Statistics
    st.subheader("Summary Statistics")
    summary_stats = pd.DataFrame({
        'Total Companies': len(filtered_df),
        'Manufacturers': filtered_df['Manufacturer_Binary'].sum(),
        'Brands': filtered_df['Brand_Binary'].sum(),
        'Distributors': filtered_df['Distributor_Binary'].sum(),
    }, index=[0])
    st.dataframe(summary_stats)
    
    # Download Section
    st.header("Download Analysis")
    
    # Prepare download data
    download_df = pd.concat([
        filtered_df,
        scores.set_index('Company')
    ], axis=1)
    
    csv = download_df.to_csv(index=False)
    st.download_button(
        label="Download Analysis as CSV",
        data=csv,
        file_name="pharma_analysis.csv",
        mime="text/csv",
    )

def main():
    st.sidebar.title("Data Upload")
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            analyzer = DataAnalyzer(df)
            create_visualizations(analyzer)
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.info("Please upload a CSV file to begin analysis")

if __name__ == "__main__":
    main()
