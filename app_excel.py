# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from io import StringIO
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Set page config
st.set_page_config(page_title="Pharma Industry Analysis", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stPlotlyChart {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

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
            self.df[f'{col}_List'] = self.df[col].str.split(',')
    
    def calculate_company_scores(self):
        # Calculate various scores for companies
        scores = pd.DataFrame()
        scores['Company'] = self.df['Company Name']
        
        # Certification Score (based on number of certifications)
        scores['Certification_Score'] = self.df['Certifications'].str.count(',') + 1
        
        # Market Presence Score
        scores['Market_Presence'] = self.df['Geographical Presence'].apply(
            lambda x: 10 if 'Global' in x else 5)
        
        # Innovation Score
        scores['Innovation_Score'] = self.df['Innovation/Research Areas'].str.count(',') + 1
        
        # Product Diversity Score
        scores['Product_Diversity'] = self.df['Product Categories'].str.count(',') + 1
        
        # Total Score
        scores['Total_Score'] = (scores['Certification_Score'] + 
                               scores['Market_Presence'] + 
                               scores['Innovation_Score'] + 
                               scores['Product_Diversity'])
        
        return scores
    
    def generate_competitor_analysis(self):
        competitor_data = []
        for _, row in self.df.iterrows():
            if row['Competitors']:
                competitors = [comp.strip() for comp in row['Competitors'].split(',')]
                for comp in competitors:
                    competitor_data.append({
                        'Company': row['Company Name'],
                        'Competitor': comp,
                        'Industry': row['Company Type']
                    })
        return pd.DataFrame(competitor_data)
    
    def analyze_market_segments(self):
        segments = {}
        for _, row in self.df.iterrows():
            if row['Health Segments']:
                for segment in row['Health Segments'].split(','):
                    segment = segment.strip()
                    if segment in segments:
                        segments[segment] += 1
                    else:
                        segments[segment] = 1
        return pd.DataFrame(list(segments.items()), columns=['Segment', 'Count'])

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
    
    # Main dashboard
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Company Type Distribution")
        fig = px.pie(filtered_df, names='Company Type', title='Company Distribution by Type')
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Manufacturing Capabilities")
        manufacturing_data = filtered_df[['Manufacturer_Binary', 'Brand_Binary', 'Distributor_Binary']].sum()
        fig = px.bar(manufacturing_data, title='Manufacturing & Distribution Capabilities')
        st.plotly_chart(fig, use_container_width=True)
    
    # Market Analysis
    st.header("Market Analysis")
    col3, col4 = st.columns(2)
    
    with col3:
        # Geographic Presence
        presence_data = filtered_df['Geographical Presence'].value_counts()
        fig = px.bar(presence_data, title='Geographic Presence Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        # Health Segments Analysis
        segments = analyzer.analyze_market_segments()
        fig = px.treemap(segments, path=['Segment'], values='Count', 
                        title='Health Segments Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    # Company Scores
    st.header("Company Performance Metrics")
    scores = analyzer.calculate_company_scores()
    fig = px.scatter(scores, x='Market_Presence', y='Innovation_Score',
                    size='Product_Diversity', hover_data=['Company'],
                    title='Company Performance Matrix')
    st.plotly_chart(fig, use_container_width=True)
    
    # Competitor Analysis
    st.header("Competitor Analysis")
    competitor_df = analyzer.generate_competitor_analysis()
    fig = px.network(competitor_df, title='Competitor Network')
    st.plotly_chart(fig, use_container_width=True)
    
    # Innovation Areas WordCloud
    st.header("Innovation Focus Areas")
    innovation_text = ' '.join(filtered_df['Innovation/Research Areas'].dropna())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(innovation_text)
    
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)
    
    # Download Section
    st.header("Download Analysis")
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="Download Data as CSV",
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
