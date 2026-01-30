#!/usr/bin/env python3
"""
================================================================================
SCRIPT 01D: This script integrates ALL available human activity data into the climate dataset:
================================================================================
1. Population 
2. CO2 Emissions (
3. NDVI 
4. Nighttime Lights 
5. Flood
6. Urban Percentage 

By Dr. Ram Chand (BNBWU)
================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_DIR = Path('processed_data')
HUMAN_ACTIVITY_DIR = Path('human_activity')

def load_current_data():
    """Load complete dataset."""
    filepath = PROCESSED_DIR / 'yearly_data_complete_22x44.csv'
    if filepath.exists():
        df = pd.read_csv(filepath)
        print(f"✓ Loaded current data: {len(df)} rows, {len(df.columns)} columns")
        return df
    else:
        print("✗ yearly_data_complete_22x44.csv not found!")
        return None

def load_nightlights():
    """Load and process nighttime lights data."""
    filepath = HUMAN_ACTIVITY_DIR / 'nightlights_combined.csv'
    if filepath.exists():
        df = pd.read_csv(filepath)
        print(f"✓ Nightlights data: {len(df)} rows ({df['year'].min()}-{df['year'].max()})")
        
        df = df.rename(columns={'nighttime_lights': 'nightlights'})
        return df[['year', 'district', 'nightlights']]
    return None

def load_flood_data():
    """Load flood occurrence data from Excel."""
    filepath = HUMAN_ACTIVITY_DIR / 'humaData' / 'flood_data.xlsx'
    if filepath.exists():
        df = pd.read_excel(filepath)
        print(f"✓ Flood data: {len(df)} rows ({df['year'].min()}-{df['year'].max()})")
       
        if 'flood_occured' in df.columns:
            df['flood_occurred'] = df['flood_occured'].fillna(0).astype(int)
        elif 'flood_occurred' in df.columns:
            df['flood_occurred'] = df['flood_occurred'].fillna(0).astype(int)
        return df[['year', 'district', 'flood_occurred']]
    return None

def load_urban_data():
    """Load urban percentage data from Excel."""
    filepath = HUMAN_ACTIVITY_DIR / 'humaData' / 'urban_data.xlsx'
    if filepath.exists():
        df = pd.read_excel(filepath)
        print(f"✓ Urban data: {len(df)} rows ({df['year'].min()}-{df['year'].max()})")
      
        df = df.rename(columns={'urban_percent': 'urban_pct_updated'})
        return df[['year', 'district', 'urban_pct_updated']]
    return None

def extend_nightlights(df, all_years, all_districts):
    """Extend nightlights data to cover all years using interpolation."""
    if df is None:
        return None

    full_grid = pd.DataFrame([(d, y) for d in all_districts for y in all_years],
                              columns=['district', 'year'])


    merged = full_grid.merge(df, on=['district', 'year'], how='left')

    result = []
    for district in all_districts:
        d_data = merged[merged['district'] == district].copy()
        d_data = d_data.sort_values('year')

      
        existing = d_data[d_data['nightlights'].notna()]

        if len(existing) > 0:
            min_year_with_data = existing['year'].min()
            max_year_with_data = existing['year'].max()

            
            d_data['nightlights'] = d_data['nightlights'].interpolate(method='linear')

            
            earliest_val = existing.iloc[0]['nightlights']
            for idx in d_data[d_data['year'] < min_year_with_data].index:
                year = d_data.loc[idx, 'year']
                years_before = min_year_with_data - year
               
                d_data.loc[idx, 'nightlights'] = earliest_val * (0.97 ** years_before)

           
            latest_val = existing.iloc[-1]['nightlights']
            for idx in d_data[d_data['year'] > max_year_with_data].index:
                year = d_data.loc[idx, 'year']
                years_after = year - max_year_with_data
                # ~2% increase per year going forward
                d_data.loc[idx, 'nightlights'] = latest_val * (1.02 ** years_after)
        else:
            
            d_data['nightlights'] = np.nan

        result.append(d_data)

    final = pd.concat(result, ignore_index=True)

    overall_mean = final['nightlights'].mean()
    final['nightlights'] = final['nightlights'].fillna(overall_mean)

    print(f"  → Extended nightlights: {len(final)} rows, coverage: {(final['nightlights'].notna().sum()/len(final)*100):.1f}%")
    return final

def extend_flood_data(df, all_years, all_districts):
    """Extend flood data to cover all years (assume 0 for missing years)."""
    if df is None:
        return None

    full_grid = pd.DataFrame([(d, y) for d in all_districts for y in all_years],
                              columns=['district', 'year'])


    merged = full_grid.merge(df, on=['district', 'year'], how='left')

  
    merged['flood_occurred'] = merged['flood_occurred'].fillna(0).astype(int)

    print(f"  → Extended flood data: {len(merged)} rows, flood years: {merged['flood_occurred'].sum()}")
    return merged

def integrate_all_data():
    """Main function to integrate all human activity data."""
    print("\n" + "=" * 70)
    print("INTEGRATING ALL HUMAN ACTIVITY DATA")
    print("=" * 70)

    main_df = load_current_data()
    if main_df is None:
        return

    all_years = sorted(main_df['year'].unique())
    all_districts = sorted(main_df['district'].unique())
    print(f"\nTarget: {len(all_districts)} districts × {len(all_years)} years = {len(all_districts)*len(all_years)} samples")

    print("\n" + "-" * 70)
    print("LOADING ADDITIONAL HUMAN ACTIVITY DATA")
    print("-" * 70)

    nightlights_df = load_nightlights()
    flood_df = load_flood_data()
    urban_df = load_urban_data()

    print("\n" + "-" * 70)
    print("EXTENDING DATA COVERAGE")
    print("-" * 70)

    if nightlights_df is not None:
        nightlights_extended = extend_nightlights(nightlights_df, all_years, all_districts)
        main_df = main_df.merge(nightlights_extended[['district', 'year', 'nightlights']],
                                 on=['district', 'year'], how='left')

    if flood_df is not None:
        flood_extended = extend_flood_data(flood_df, all_years, all_districts)
        main_df = main_df.merge(flood_extended[['district', 'year', 'flood_occurred']],
                                 on=['district', 'year'], how='left')

    if urban_df is not None:
        main_df = main_df.merge(urban_df, on=['district', 'year'], how='left')
        if 'urban_pct_updated' in main_df.columns:
            mask = main_df['urban_pct_updated'].notna()
            main_df.loc[mask, 'urban_pct'] = main_df.loc[mask, 'urban_pct_updated']
            main_df = main_df.drop(columns=['urban_pct_updated'])

    print("\n" + "-" * 70)
    print("CALCULATING DERIVED FEATURES")
    print("-" * 70)

  
    main_df = main_df.sort_values(['district', 'year'])
    main_df['cumulative_floods'] = main_df.groupby('district')['flood_occurred'].cumsum()
    print(f"  ✓ Added cumulative_floods (total flood events per district over time)")

   
    if 'co2_per_capita_kg' not in main_df.columns:
        if 'co2_emissions_kt' in main_df.columns and 'population' in main_df.columns:
            main_df['co2_per_capita_kg'] = (main_df['co2_emissions_kt'] * 1e6) / main_df['population']
            print(f"  ✓ Added co2_per_capita_kg")

    
    main_df['nightlights_change'] = main_df.groupby('district')['nightlights'].pct_change()
    main_df['nightlights_change'] = main_df['nightlights_change'].fillna(0)
    print(f"  ✓ Added nightlights_change (year-over-year change)")

    
    print("\n" + "=" * 70)
    print("INTEGRATION COMPLETE")
    print("=" * 70)

    human_activity_cols = ['population', 'pop_density', 'urban_pct', 'co2_emissions_kt',
                           'national_co2_mt', 'co2_per_capita_kg', 'ndvi_mean',
                           'nightlights', 'nightlights_change', 'flood_occurred', 'cumulative_floods']

    available_cols = [c for c in human_activity_cols if c in main_df.columns]

    print(f"\nHuman Activity Features ({len(available_cols)}):")
    for col in available_cols:
        non_null = main_df[col].notna().sum()
        coverage = non_null / len(main_df) * 100
        print(f"  • {col}: {coverage:.1f}% coverage")

    output_path = PROCESSED_DIR / 'yearly_data_complete_22x44_v2.csv'
    main_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved: {output_path}")
    print(f"  Shape: {main_df.shape}")

    main_path = PROCESSED_DIR / 'yearly_data_complete_22x44.csv'
    main_df.to_csv(main_path, index=False)
    print(f"✓ Updated: {main_path}")

    return main_df

if __name__ == "__main__":
    df = integrate_all_data()
