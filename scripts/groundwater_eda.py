#!/usr/bin/env python3
"""
Comprehensive EDA for Bengaluru Groundwater Dataset (2010-2025)
This script performs detailed exploratory data analysis on 15 years of groundwater data.
"""

import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

class GroundwaterEDA:
    def __init__(self, dataset_path="dataset"):
        self.dataset_path = dataset_path
        self.df = None
        self.stations_info = None
        
    def load_all_data(self):
        """Load all JSON files from the dataset directory"""
        print("Loading groundwater data from all files...")
        
        all_data = []
        file_count = 0
        
        # Get all JSON files in the dataset directory
        import os
        json_files = [f for f in os.listdir(self.dataset_path) if f.endswith('.json')]
        json_files.sort()  # Sort by filename to maintain chronological order
        
        for filename in json_files:
            file_path = os.path.join(self.dataset_path, filename)
            print(f"Loading {filename}...")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    all_data.extend(data)
                    file_count += 1
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue
        
        print(f"Successfully loaded {file_count} files with {len(all_data)} total records")
        
        # Convert to DataFrame
        self.df = pd.DataFrame(all_data)
        return self.df
    
    def clean_data(self):
        """Clean and preprocess the data"""
        print("\nCleaning and preprocessing data...")
        
        if self.df is None:
            print("No data loaded. Please run load_all_data() first.")
            return
        
        # Create a copy for cleaning
        df_clean = self.df.copy()
        
        # Convert dataTime to datetime
        print("Converting timestamps...")
        df_clean['datetime'] = pd.to_datetime(
            df_clean['dataTime'].apply(lambda x: f"{x['year']}-{x['monthValue']:02d}-{x['dayOfMonth']:02d} {x['hour']:02d}:{x['minute']:02d}:{x['second']:02d}")
        )
        
        # Extract useful time features
        df_clean['year'] = df_clean['datetime'].dt.year
        df_clean['month'] = df_clean['datetime'].dt.month
        df_clean['day'] = df_clean['datetime'].dt.day
        df_clean['hour'] = df_clean['datetime'].dt.hour
        df_clean['quarter'] = df_clean['datetime'].dt.quarter
        df_clean['day_of_year'] = df_clean['datetime'].dt.dayofyear
        
        # Handle missing values in critical columns
        print("Handling missing values...")
        df_clean['dataValue'] = pd.to_numeric(df_clean['dataValue'], errors='coerce')
        df_clean['latitude'] = pd.to_numeric(df_clean['latitude'], errors='coerce')
        df_clean['longitude'] = pd.to_numeric(df_clean['longitude'], errors='coerce')
        df_clean['wellDepth'] = pd.to_numeric(df_clean['wellDepth'], errors='coerce')
        
        # Remove records with missing critical data
        initial_count = len(df_clean)
        df_clean = df_clean.dropna(subset=['dataValue', 'latitude', 'longitude', 'datetime'])
        final_count = len(df_clean)
        
        print(f"Removed {initial_count - final_count} records with missing critical data")
        
        # Create station identifier
        df_clean['station_id'] = df_clean['stationCode'] + '_' + df_clean['stationName']
        
        # Sort by datetime
        df_clean = df_clean.sort_values('datetime')
        
        self.df = df_clean
        print(f"Final dataset shape: {self.df.shape}")
        
        return self.df
    
    def analyze_data_structure(self):
        """Analyze the structure and quality of the data"""
        print("\n" + "="*60)
        print("DATA STRUCTURE ANALYSIS")
        print("="*60)
        
        if self.df is None:
            print("No data loaded. Please run load_all_data() and clean_data() first.")
            return
        
        # Basic info
        print(f"Dataset shape: {self.df.shape}")
        print(f"Date range: {self.df['datetime'].min()} to {self.df['datetime'].max()}")
        print(f"Number of unique stations: {self.df['stationCode'].nunique()}")
        print(f"Number of unique station names: {self.df['stationName'].nunique()}")
        
        # Data types
        print("\nData types:")
        print(self.df.dtypes)
        
        # Missing values
        print("\nMissing values:")
        missing_data = self.df.isnull().sum()
        missing_percent = (missing_data / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing_data,
            'Missing Percent': missing_percent
        })
        print(missing_df[missing_df['Missing Count'] > 0])
        
        # Unique values in categorical columns
        print("\nUnique values in categorical columns:")
        categorical_cols = ['stationType', 'agencyName', 'state', 'district', 'tehsil', 
                          'datatypeCode', 'description', 'unit', 'wellType', 'wellAquiferType']
        
        for col in categorical_cols:
            if col in self.df.columns:
                unique_count = self.df[col].nunique()
                print(f"{col}: {unique_count} unique values")
                if unique_count <= 10:
                    print(f"  Values: {self.df[col].unique()}")
    
    def analyze_stations(self):
        """Analyze station information and distribution"""
        print("\n" + "="*60)
        print("STATION ANALYSIS")
        print("="*60)
        
        if self.df is None:
            print("No data loaded. Please run load_all_data() and clean_data() first.")
            return
        
        # Station summary
        station_summary = self.df.groupby(['stationCode', 'stationName']).agg({
            'datetime': ['count', 'min', 'max'],
            'latitude': 'first',
            'longitude': 'first',
            'wellDepth': 'first',
            'wellType': 'first',
            'wellAquiferType': 'first',
            'dataValue': ['mean', 'std', 'min', 'max']
        }).round(2)
        
        station_summary.columns = ['record_count', 'first_date', 'last_date', 
                                 'latitude', 'longitude', 'well_depth', 'well_type',
                                 'aquifer_type', 'mean_water_level', 'std_water_level',
                                 'min_water_level', 'max_water_level']
        
        print("Station Summary:")
        print(station_summary)
        
        # Save station summary
        station_summary.to_csv('station_summary.csv')
        print("\nStation summary saved to 'station_summary.csv'")
        
        # Station distribution by type
        print("\nStation distribution by well type:")
        well_type_dist = self.df.groupby('wellType')['stationCode'].nunique()
        print(well_type_dist)
        
        print("\nStation distribution by aquifer type:")
        aquifer_dist = self.df.groupby('wellAquiferType')['stationCode'].nunique()
        print(aquifer_dist)
        
        # Geographic distribution
        print(f"\nGeographic bounds:")
        print(f"Latitude: {self.df['latitude'].min():.4f} to {self.df['latitude'].max():.4f}")
        print(f"Longitude: {self.df['longitude'].min():.4f} to {self.df['longitude'].max():.4f}")
        
        return station_summary
    
    def analyze_water_levels(self):
        """Analyze water level data"""
        print("\n" + "="*60)
        print("WATER LEVEL ANALYSIS")
        print("="*60)
        
        if self.df is None:
            print("No data loaded. Please run load_all_data() and clean_data() first.")
            return
        
        # Basic statistics
        print("Water Level Statistics (in meters):")
        stats = self.df['dataValue'].describe()
        print(stats)
        
        # Check for negative values (indicating above ground level)
        negative_count = (self.df['dataValue'] < 0).sum()
        negative_percent = (negative_count / len(self.df)) * 100
        print(f"\nNegative water levels (above ground): {negative_count} ({negative_percent:.2f}%)")
        
        # Water level distribution by well type
        print("\nWater level statistics by well type:")
        well_type_stats = self.df.groupby('wellType')['dataValue'].describe()
        print(well_type_stats)
        
        # Water level distribution by aquifer type
        print("\nWater level statistics by aquifer type:")
        aquifer_stats = self.df.groupby('wellAquiferType')['dataValue'].describe()
        print(aquifer_stats)
        
        # Temporal analysis
        print("\nWater level statistics by year:")
        yearly_stats = self.df.groupby('year')['dataValue'].describe()
        print(yearly_stats)
        
        return stats
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("\n" + "="*60)
        print("CREATING VISUALIZATIONS")
        print("="*60)
        
        if self.df is None:
            print("No data loaded. Please run load_all_data() and clean_data() first.")
            return
        
        try:
            # Set up the plotting style
            plt.rcParams['figure.figsize'] = (12, 8)
            
            # 1. Water Level Distribution
            plt.figure(figsize=(15, 10))
            
            plt.subplot(2, 3, 1)
            plt.hist(self.df['dataValue'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            plt.title('Distribution of Water Levels')
            plt.xlabel('Water Level (m)')
            plt.ylabel('Frequency')
            plt.axvline(self.df['dataValue'].mean(), color='red', linestyle='--', label=f'Mean: {self.df["dataValue"].mean():.2f}m')
            plt.legend()
            
            # 2. Water Levels Over Time
            plt.subplot(2, 3, 2)
            yearly_avg = self.df.groupby('year')['dataValue'].mean()
            plt.plot(yearly_avg.index, yearly_avg.values, marker='o', linewidth=2, markersize=6)
            plt.title('Average Water Levels by Year')
            plt.xlabel('Year')
            plt.ylabel('Average Water Level (m)')
            plt.grid(True, alpha=0.3)
            
            # 3. Monthly Patterns
            plt.subplot(2, 3, 3)
            monthly_avg = self.df.groupby('month')['dataValue'].mean()
            plt.bar(monthly_avg.index, monthly_avg.values, color='lightgreen', alpha=0.7)
            plt.title('Average Water Levels by Month')
            plt.xlabel('Month')
            plt.ylabel('Average Water Level (m)')
            plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
            
            # 4. Station Distribution Map
            plt.subplot(2, 3, 4)
            station_coords = self.df.groupby('stationCode')[['latitude', 'longitude']].first()
            plt.scatter(station_coords['longitude'], station_coords['latitude'], 
                       alpha=0.6, s=50, c='red')
            plt.title('Station Locations')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.grid(True, alpha=0.3)
            
            # 5. Water Level by Well Type
            plt.subplot(2, 3, 5)
            well_type_data = [self.df[self.df['wellType'] == well_type]['dataValue'].values 
                             for well_type in self.df['wellType'].unique()]
            plt.boxplot(well_type_data, labels=self.df['wellType'].unique())
            plt.title('Water Levels by Well Type')
            plt.xlabel('Well Type')
            plt.ylabel('Water Level (m)')
            plt.xticks(rotation=45)
            
            # 6. Data Volume Over Time
            plt.subplot(2, 3, 6)
            monthly_counts = self.df.groupby([self.df['datetime'].dt.year, self.df['datetime'].dt.month]).size()
            monthly_counts.index = [f"{year}-{month:02d}" for year, month in monthly_counts.index]
            plt.plot(range(len(monthly_counts)), monthly_counts.values, marker='o', alpha=0.7)
            plt.title('Data Volume Over Time')
            plt.xlabel('Time Period')
            plt.ylabel('Number of Records')
            plt.xticks(range(0, len(monthly_counts), 12), monthly_counts.index[::12], rotation=45)
            
            plt.tight_layout()
            plt.savefig('groundwater_analysis_overview.png', dpi=300, bbox_inches='tight')
            print("✓ Saved groundwater_analysis_overview.png")
            plt.close()
            
            # Additional detailed plots
            self._create_detailed_plots()
            
        except Exception as e:
            print(f"Error creating visualizations: {e}")
            print("Continuing with other analyses...")
    
    def _create_detailed_plots(self):
        """Create additional detailed visualizations"""
        try:
            # 1. Time series for individual stations
            plt.figure(figsize=(15, 10))
            
            # Get top 5 stations by data volume
            top_stations = self.df.groupby('stationCode').size().nlargest(5).index
            
            for i, station in enumerate(top_stations):
                station_data = self.df[self.df['stationCode'] == station]
                station_name = station_data['stationName'].iloc[0]
                
                plt.subplot(2, 3, i+1)
                plt.scatter(station_data['datetime'], station_data['dataValue'], 
                           alpha=0.6, s=20)
                plt.title(f'{station_name}\n({station})')
                plt.xlabel('Date')
                plt.ylabel('Water Level (m)')
                plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig('station_time_series.png', dpi=300, bbox_inches='tight')
            print("✓ Saved station_time_series.png")
            plt.close()
            
            # 2. Seasonal patterns
            plt.figure(figsize=(12, 8))
            
            # Create seasonal data
            self.df['season'] = pd.cut(self.df['month'], 
                                      bins=[0, 3, 6, 9, 12], 
                                      labels=['Winter', 'Spring', 'Summer', 'Fall'])
            
            seasonal_avg = self.df.groupby(['year', 'season'])['dataValue'].mean().unstack()
            
            seasonal_avg.plot(kind='line', marker='o', figsize=(12, 6))
            plt.title('Seasonal Water Level Patterns')
            plt.xlabel('Year')
            plt.ylabel('Average Water Level (m)')
            plt.legend(title='Season')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('seasonal_patterns.png', dpi=300, bbox_inches='tight')
            print("✓ Saved seasonal_patterns.png")
            plt.close()
            
            # 3. Correlation heatmap
            plt.figure(figsize=(10, 8))
            
            # Select numeric columns for correlation
            numeric_cols = ['dataValue', 'latitude', 'longitude', 'wellDepth', 
                           'year', 'month', 'day', 'hour', 'day_of_year']
            correlation_data = self.df[numeric_cols].corr()
            
            sns.heatmap(correlation_data, annot=True, cmap='coolwarm', center=0,
                       square=True, linewidths=0.5)
            plt.title('Correlation Matrix')
            plt.tight_layout()
            plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
            print("✓ Saved correlation_heatmap.png")
            plt.close()
            
        except Exception as e:
            print(f"Error creating detailed plots: {e}")
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        print("\n" + "="*60)
        print("GENERATING SUMMARY REPORT")
        print("="*60)
        
        if self.df is None:
            print("No data loaded. Please run load_all_data() and clean_data() first.")
            return
        
        # Create summary statistics
        summary_stats = {
            'Total Records': len(self.df),
            'Date Range': f"{self.df['datetime'].min().strftime('%Y-%m-%d')} to {self.df['datetime'].max().strftime('%Y-%m-%d')}",
            'Number of Years': self.df['year'].nunique(),
            'Number of Stations': self.df['stationCode'].nunique(),
            'Average Water Level': f"{self.df['dataValue'].mean():.2f} m",
            'Water Level Range': f"{self.df['dataValue'].min():.2f} to {self.df['dataValue'].max():.2f} m",
            'Standard Deviation': f"{self.df['dataValue'].std():.2f} m",
            'Well Types': list(self.df['wellType'].unique()),
            'Aquifer Types': list(self.df['wellAquiferType'].unique()),
            'Geographic Bounds': {
                'Latitude': f"{self.df['latitude'].min():.4f} to {self.df['latitude'].max():.4f}",
                'Longitude': f"{self.df['longitude'].min():.4f} to {self.df['longitude'].max():.4f}"
            }
        }
        
        # Save summary to file
        with open('groundwater_summary_report.txt', 'w') as f:
            f.write("BENGALURU GROUNDWATER DATASET SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            for key, value in summary_stats.items():
                f.write(f"{key}: {value}\n")
            
            f.write("\n" + "=" * 50 + "\n")
            f.write("DATA QUALITY ASSESSMENT\n")
            f.write("=" * 50 + "\n")
            
            # Data quality metrics
            f.write(f"Missing values in critical columns:\n")
            missing_data = self.df[['dataValue', 'latitude', 'longitude', 'datetime']].isnull().sum()
            for col, count in missing_data.items():
                f.write(f"  {col}: {count} ({count/len(self.df)*100:.2f}%)\n")
            
            # Check for duplicates excluding the dataTime column which contains dictionaries
            df_for_duplicates = self.df.drop(columns=['dataTime'])
            f.write(f"\nDuplicate records: {(df_for_duplicates.duplicated().sum())}\n")
            f.write(f"Unique station-stationName combinations: {self.df.groupby(['stationCode', 'stationName']).ngroups}\n")
        
        print("✓ Summary report saved to 'groundwater_summary_report.txt'")
        
        return summary_stats
    
    def run_complete_eda(self):
        """Run the complete EDA pipeline"""
        print("BENGALURU GROUNDWATER DATASET - COMPREHENSIVE EDA")
        print("=" * 60)
        
        # Load data
        self.load_all_data()
        
        # Clean data
        self.clean_data()
        
        # Run all analyses
        self.analyze_data_structure()
        self.analyze_stations()
        self.analyze_water_levels()
        self.create_visualizations()
        self.generate_summary_report()
        
        print("\n" + "="*60)
        print("EDA COMPLETE!")
        print("="*60)
        print("Generated files:")
        print("- groundwater_analysis_overview.png")
        print("- station_time_series.png") 
        print("- seasonal_patterns.png")
        print("- correlation_heatmap.png")
        print("- station_summary.csv")
        print("- groundwater_summary_report.txt")

if __name__ == "__main__":
    # Initialize and run EDA
    eda = GroundwaterEDA()
    eda.run_complete_eda() 