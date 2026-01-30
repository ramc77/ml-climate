#!/usr/bin/env python3
"""
================================================================================
SCRIPT 05: VISUALIZATION
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


PROCESSED_DIR = Path('processed_data')
RESULTS_DIR = Path('analysis_results')
FIGURES_DIR = Path('figures')

DPI = 300
FIGSIZE_WIDE = (14, 8)
FIGSIZE_SQUARE = (10, 10)
FIGSIZE_TALL = (12, 14)

# Create output directories
FOLDERS = {
    'A': 'A_Climate_Trends',
    'B': 'B_Hotspot_Analysis',
    'C': 'C_Model_Performance',
    'D': 'D_Attribution',
    'E': 'E_Paris_Agreement',
    'F': 'F_IPCC_Risk',
    'G': 'G_Projections',
    'H': 'H_Policy',
    'I': 'I_Summary',
}

for folder in FOLDERS.values():
    (FIGURES_DIR / folder).mkdir(parents=True, exist_ok=True)

# District coordinates
SINDH_COORDS = {
    'Karachi': (24.86, 67.00), 'Hyderabad': (25.40, 68.36), 'Sukkur': (27.71, 68.86),
    'Larkana': (27.56, 68.21), 'Nawabshah': (26.25, 68.40), 'Mirpurkhas': (25.53, 69.01),
    'Thatta': (24.75, 67.92), 'Badin': (24.63, 68.84), 'Tharparkar': (24.74, 70.13),
    'Umerkot': (25.36, 69.74), 'Sanghar': (26.05, 68.95), 'Khairpur': (27.53, 68.76),
    'Ghotki': (28.00, 69.32), 'Jacobabad': (28.28, 68.44), 'Shikarpur': (27.96, 68.64),
    'Kashmore': (28.43, 69.58), 'Dadu': (26.73, 67.78), 'Jamshoro': (25.43, 68.28),
    'Matiari': (25.59, 68.45), 'Tando Allahyar': (25.46, 68.72),
    'Tando Muhammad Khan': (25.12, 68.53), 'Sujawal': (24.56, 68.02),
}

RISK_COLORS = {
    'Critical': '#B71C1C', 'Very High': '#D32F2F', 'High': '#F57C00',
    'Medium': '#FBC02D', 'Moderate': '#FBC02D', 'Low': '#388E3C',
    'Very Low': '#1B5E20', 'Very Low Risk': '#1B5E20',
    'Low Risk': '#388E3C', 'Medium Risk': '#FBC02D',
    'High Risk': '#F57C00', 'Very High Risk': '#D32F2F',
}


def load_v4_data():
    """Load all data files."""
    print("\n" + "=" * 70)
    print("LOADING DATA FILES")
    print("=" * 70)
    
    data = {}
    
    v4_files = {
        'lodo': RESULTS_DIR / 'lodo_cv.csv',
        'ml_results': RESULTS_DIR / 'ml_results.csv',
        'predictions': RESULTS_DIR / 'predictions.csv',
        'attribution': RESULTS_DIR / 'attribution_importance.csv',
        'attribution_categories': RESULTS_DIR / 'attribution_categories.csv',
        'trends': RESULTS_DIR / 'climate_trends.csv',
        'hotspot': RESULTS_DIR / 'hotspot_analysis.csv',
        'ranking': RESULTS_DIR / 'comprehensive_ranking.csv',
        'paris': RESULTS_DIR / 'paris_compliance.csv',
        'ipcc': RESULTS_DIR / 'ipcc_risk.csv',
        'projections': RESULTS_DIR / 'climate_projections.csv',
    }
    
    processed_files = {
        'full': PROCESSED_DIR / 'full_processed.csv',
        'train': PROCESSED_DIR / 'train.csv',
        'test': PROCESSED_DIR / 'test.csv',
    }
    
    for name, path in v4_files.items():
        if path.exists():
            data[name] = pd.read_csv(path)
            print(f"  ✓ {name}: {len(data[name])} rows")
        else:
            print(f"  ✗ {name}: NOT FOUND at {path}")
    
    for name, path in processed_files.items():
        if path.exists():
            data[name] = pd.read_csv(path)
            print(f"  ✓ {name}: {len(data[name])} rows")
        else:
            print(f"  ✗ {name}: NOT FOUND")
    
    return data

def save_fig(fig, folder, filename):
    """Save figures."""
    filepath = FIGURES_DIR / folder / filename
    fig.savefig(filepath, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"    ✓ {filename}")
    return 1


def plot_section_A(data):
    """Climate trends visualizations."""
    print("\n" + "=" * 70)
    print("SECTION A: CLIMATE TRENDS")
    print("=" * 70)
    
    folder = FOLDERS['A']
    count = 0
    
    if 'full' not in data or 'trends' not in data:
        print("  ✗ Required data not available")
        return 0
    
    full = data['full']
    trends = data['trends']
    
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    
    yearly_mean = full.groupby('year')['T2M_mean'].mean()
    yearly_std = full.groupby('year')['T2M_mean'].std()
    
    ax.fill_between(yearly_mean.index, yearly_mean - yearly_std, yearly_mean + yearly_std, alpha=0.3)
    ax.plot(yearly_mean.index, yearly_mean, 'b-o', linewidth=2, markersize=4, label='Mean ± Std')
    
    z = np.polyfit(yearly_mean.index, yearly_mean.values, 1)
    ax.plot(yearly_mean.index, np.poly1d(z)(yearly_mean.index), 'r--', linewidth=2,
           label=f'Trend: {z[0]*10:.3f}°C/decade')
    
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Temperature (°C)', fontsize=12)
    ax.set_title('Mean Temperature Trends in Sindh Province (1981-2024)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    count += save_fig(fig, folder, 'A01_temperature_timeseries.png')
    
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    
    trends_sorted = trends.sort_values('trend_per_decade', ascending=True)
    colors = ['green' if t < 0 else 'red' for t in trends_sorted['trend_per_decade']]
    
    ax.barh(trends_sorted['district'], trends_sorted['trend_per_decade'], color=colors, edgecolor='black')
    ax.axvline(x=0, color='black', linewidth=2)
    ax.set_xlabel('Temperature Trend (°C/decade)', fontsize=12)
    ax.set_title('Warming Rates by District', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    count += save_fig(fig, folder, 'A02_district_warming_rates.png')
    
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    
    karachi = full[full['district'] == 'Karachi'].groupby('year')['T2M_mean'].mean()
    others = full[full['district'] != 'Karachi'].groupby('year')['T2M_mean'].mean()
    
    ax.plot(karachi.index, karachi.values, 'r-o', linewidth=2, markersize=4, label='Karachi')
    ax.plot(others.index, others.values, 'b-s', linewidth=2, markersize=4, label='Other Districts (Mean)')
    
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Temperature (°C)', fontsize=12)
    ax.set_title('Karachi vs Other Districts: Temperature Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    count += save_fig(fig, folder, 'A03_karachi_comparison.png')
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    pivot = full.pivot_table(values='temp_anomaly', index='district', columns='year', aggfunc='mean')
    
    im = ax.imshow(pivot.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    ax.set_xticks(range(0, len(pivot.columns), 5))
    ax.set_xticklabels(pivot.columns[::5], rotation=45)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel('Year')
    ax.set_ylabel('District')
    ax.set_title('Temperature Anomaly Heatmap (°C from baseline)', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Anomaly (°C)')
    
    plt.tight_layout()
    count += save_fig(fig, folder, 'A04_anomaly_heatmap.png')
    
    print(f"  Section A complete: {count} figures")
    return count


def plot_section_B(data):
    """Hotspot analysis visualizations."""
    print("\n" + "=" * 70)
    print("SECTION B: HOTSPOT ANALYSIS")
    print("=" * 70)
    
    folder = FOLDERS['B']
    count = 0
    
    if 'hotspot' not in data or 'ranking' not in data:
        print("  ✗ Required data not available")
        return 0
    
    hotspot = data['hotspot']
    ranking = data['ranking']
    
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    
    ranking_sorted = ranking.sort_values('hotspot_score', ascending=True)
    colors = [RISK_COLORS.get(level, '#888888') for level in ranking_sorted['priority_level']]
    
    ax.barh(ranking_sorted['district'], ranking_sorted['hotspot_score'], color=colors, edgecolor='black')
    ax.set_xlabel('Hotspot Score', fontsize=12)
    ax.set_title('Climate Hotspot Ranking', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.1)
    
    handles = [mpatches.Patch(color=RISK_COLORS[level], label=level) 
               for level in ['Critical', 'High', 'Medium', 'Low'] if level in RISK_COLORS]
    ax.legend(handles=handles, title='Priority Level', loc='lower right')
    
    count += save_fig(fig, folder, 'B01_hotspot_ranking.png')
    
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    
    components = ['hazard_index', 'exposure_index', 'vulnerability_index']
    x = np.arange(len(hotspot))
    width = 0.25
    
    hotspot_sorted = hotspot.sort_values('hotspot_score', ascending=False)
    
    for i, comp in enumerate(components):
        if comp in hotspot_sorted.columns:
            ax.bar(x + i*width, hotspot_sorted[comp], width, label=comp.replace('_', ' ').title())
    
    ax.set_xticks(x + width)
    ax.set_xticklabels(hotspot_sorted['district'], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Index Value')
    ax.set_title('Risk Components by District (IPCC Framework)', fontsize=14, fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    count += save_fig(fig, folder, 'B02_risk_components.png')
    
    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    
    for _, row in hotspot.iterrows():
        district = row['district']
        if district in SINDH_COORDS:
            lat, lon = SINDH_COORDS[district]
            score = row['hotspot_score']
            color = plt.cm.Reds(score)
            size = 200 + score * 300
            ax.scatter(lon, lat, c=[color], s=size, alpha=0.7, edgecolors='black')
            ax.text(lon, lat, f'{score:.2f}', ha='center', va='center', fontsize=7, fontweight='bold')
    
    ax.set_xlim(66, 71)
    ax.set_ylim(23.5, 29)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Hotspot Score Spatial Distribution', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    sm = plt.cm.ScalarMappable(cmap='Reds', norm=Normalize(0, 1))
    plt.colorbar(sm, ax=ax, label='Hotspot Score', shrink=0.7)
    
    count += save_fig(fig, folder, 'B03_hotspot_map.png')
    
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    
    top10 = ranking.head(10)
    
    ax.barh(range(10), top10['hotspot_score'], color='coral', edgecolor='black')
    ax.set_yticks(range(10))
    ax.set_yticklabels([f"{i+1}. {d}" for i, d in enumerate(top10['district'])])
    ax.set_xlabel('Hotspot Score')
    ax.set_title('Top 10 Climate Hotspots in Sindh', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    for i, (_, row) in enumerate(top10.iterrows()):
        ax.text(row['hotspot_score'] + 0.02, i, f"R²={row['model_r2']:.2f}", va='center', fontsize=9)
    
    count += save_fig(fig, folder, 'B04_top10_hotspots.png')
    
    print(f"  Section B complete: {count} figures")
    return count


def plot_section_C(data):
    """Model performance visualizations."""
    print("\n" + "=" * 70)
    print("SECTION C: MODEL PERFORMANCE")
    print("=" * 70)
    
    folder = FOLDERS['C']
    count = 0
    
    if 'lodo' not in data:
        print("  ✗ Required data not available")
        return 0
    
    lodo = data['lodo']
    
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    
    lodo_sorted = lodo.sort_values('r2', ascending=True)
    colors = ['green' if r2 >= 0.9 else 'orange' if r2 >= 0.7 else 'red' for r2 in lodo_sorted['r2']]
    
    ax.barh(lodo_sorted['district'], lodo_sorted['r2'], color=colors, edgecolor='black')
    ax.axvline(x=0.9, color='green', linestyle='--', linewidth=2, label='Excellent (0.9)')
    ax.axvline(x=0.7, color='orange', linestyle='--', linewidth=2, label='Good (0.7)')
    
    ax.set_xlabel('R² Score', fontsize=12)
    ax.set_title(f'LODO-CV Performance by District (Mean R² = {lodo["r2"].mean():.3f})', 
                fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.05)
    ax.legend(loc='lower right')
    
    count += save_fig(fig, folder, 'C01_lodo_cv_r2.png')
    
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    
    ax.hist(lodo['r2'], bins=15, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(x=lodo['r2'].mean(), color='red', linewidth=2, label=f'Mean: {lodo["r2"].mean():.3f}')
    ax.axvline(x=lodo['r2'].median(), color='orange', linewidth=2, label=f'Median: {lodo["r2"].median():.3f}')
    
    ax.set_xlabel('R² Score', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of LODO-CV R² Scores', fontsize=14, fontweight='bold')
    ax.legend()
    
    count += save_fig(fig, folder, 'C02_r2_distribution.png')
    
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    
    lodo_sorted = lodo.sort_values('rmse', ascending=False)
    ax.barh(lodo_sorted['district'], lodo_sorted['rmse'], color='coral', edgecolor='black')
    ax.set_xlabel('RMSE (°C)', fontsize=12)
    ax.set_title('Model RMSE by District', fontsize=14, fontweight='bold')
    
    count += save_fig(fig, folder, 'C03_rmse_by_district.png')
    
    if 'predictions' in data:
        pred = data['predictions']
        
        fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
        
        ax.scatter(pred['y_true'], pred['y_pred'], alpha=0.5, c='steelblue', edgecolors='black')
        
        min_val = min(pred['y_true'].min(), pred['y_pred'].min())
        max_val = max(pred['y_true'].max(), pred['y_pred'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        r2 = 1 - np.sum((pred['y_true'] - pred['y_pred'])**2) / np.sum((pred['y_true'] - pred['y_true'].mean())**2)
        ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, fontsize=12,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white'))
        
        ax.set_xlabel('Actual Temperature Anomaly (°C)', fontsize=12)
        ax.set_ylabel('Predicted Temperature Anomaly (°C)', fontsize=12)
        ax.set_title('Predictions vs Actual (Test Set)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        count += save_fig(fig, folder, 'C04_predictions_scatter.png')
    
    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    
    for _, row in lodo.iterrows():
        district = row['district']
        if district in SINDH_COORDS:
            lat, lon = SINDH_COORDS[district]
            r2 = row['r2']
            color = plt.cm.RdYlGn(r2)
            ax.scatter(lon, lat, c=[color], s=300, alpha=0.8, edgecolors='black')
            ax.text(lon, lat, f'{r2:.2f}', ha='center', va='center', fontsize=7, fontweight='bold')
    
    ax.set_xlim(66, 71)
    ax.set_ylim(23.5, 29)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('LODO-CV R² Spatial Distribution', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=Normalize(0.5, 1))
    plt.colorbar(sm, ax=ax, label='R²', shrink=0.7)
    
    count += save_fig(fig, folder, 'C05_lodo_spatial.png')
    
    print(f"  Section C complete: {count} figures")
    return count


def plot_section_D(data):
    """Attribution analysis visualizations."""
    print("\n" + "=" * 70)
    print("SECTION D: ATTRIBUTION ANALYSIS")
    print("=" * 70)
    
    folder = FOLDERS['D']
    count = 0
    
    if 'attribution' not in data:
        print("  ✗ Required data not available")
        return 0
    
    attr = data['attribution']
    
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    
    attr_sorted = attr.sort_values('importance_mean', ascending=True).tail(15)
    
    ax.barh(attr_sorted['feature'], attr_sorted['importance_mean'], 
           xerr=attr_sorted['importance_std'], color='steelblue', edgecolor='black', capsize=3)
    ax.set_xlabel('Importance (Permutation)', fontsize=12)
    ax.set_title('Top 15 Feature Importance', fontsize=14, fontweight='bold')
    
    count += save_fig(fig, folder, 'D01_feature_importance.png')
    
    if 'attribution_categories' in data:
        cat = data['attribution_categories']
        
        fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(cat)))
        ax.pie(cat['contribution_pct'], labels=cat['category'], colors=colors,
              autopct='%1.1f%%', startangle=90)
        ax.set_title('Attribution by Category', fontsize=14, fontweight='bold')
        
        count += save_fig(fig, folder, 'D02_category_pie.png')
    
    print(f"  Section D complete: {count} figures")
    return count

def plot_section_E(data):
    """Paris Agreement analysis visualizations."""
    print("\n" + "=" * 70)
    print("SECTION E: PARIS AGREEMENT")
    print("=" * 70)
    
    folder = FOLDERS['E']
    count = 0
    
    if 'paris' not in data:
        print("  ✗ Required data not available")
        return 0
    
    paris = data['paris']
    
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    
    status_counts = paris['compliance_status'].value_counts()
    colors = ['green' if 'Track' in s else 'orange' if 'Risk' in s else 'red' for s in status_counts.index]
    
    ax.bar(status_counts.index, status_counts.values, color=colors, edgecolor='black')
    ax.set_ylabel('Number of Districts', fontsize=12)
    ax.set_title('Paris Agreement Compliance Status', fontsize=14, fontweight='bold')
    
    for i, v in enumerate(status_counts.values):
        ax.text(i, v + 0.5, str(v), ha='center', fontsize=12, fontweight='bold')
    
    count += save_fig(fig, folder, 'E01_compliance_status.png')
    
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    
    paris_sorted = paris.sort_values('warming_rate', ascending=True)
    colors = ['red' if wr > 0 else 'blue' for wr in paris_sorted['warming_rate']]
    
    ax.barh(paris_sorted['district'], paris_sorted['warming_rate'] * 10, color=colors, edgecolor='black')
    ax.axvline(x=0, color='black', linewidth=2)
    ax.set_xlabel('Warming Rate (°C/decade)', fontsize=12)
    ax.set_title('District Warming Rates', fontsize=14, fontweight='bold')
    
    count += save_fig(fig, folder, 'E02_warming_rates.png')
    
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    
    paris_filtered = paris[paris['years_to_1_5C'] < 500].sort_values('years_to_1_5C')
    
    if len(paris_filtered) > 0:
        colors = ['red' if y < 30 else 'orange' if y < 100 else 'green' for y in paris_filtered['years_to_1_5C']]
        ax.barh(paris_filtered['district'], paris_filtered['years_to_1_5C'], color=colors, edgecolor='black')
        ax.axvline(x=26, color='red', linestyle='--', linewidth=2, label='2050')
        ax.axvline(x=56, color='orange', linestyle='--', linewidth=2, label='2080')
        ax.set_xlabel('Years to 1.5°C Threshold', fontsize=12)
        ax.set_title('Time to Paris Agreement 1.5°C Threshold', fontsize=14, fontweight='bold')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'Most districts show cooling trends\nNo imminent threshold crossing', 
               transform=ax.transAxes, ha='center', va='center', fontsize=14)
        ax.set_title('Paris Agreement Analysis', fontsize=14, fontweight='bold')
    
    count += save_fig(fig, folder, 'E03_years_to_threshold.png')
    
    print(f"  Section E complete: {count} figures")
    return count

def plot_section_F(data):
    """IPCC risk framework visualizations."""
    print("\n" + "=" * 70)
    print("SECTION F: IPCC RISK FRAMEWORK")
    print("=" * 70)
    
    folder = FOLDERS['F']
    count = 0
    
    if 'ipcc' not in data:
        print("  ✗ Required data not available")
        return 0
    
    ipcc = data['ipcc']
    
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    
    level_counts = ipcc['ipcc_level'].value_counts()
    colors = [RISK_COLORS.get(level, '#888888') for level in level_counts.index]
    
    ax.bar(level_counts.index, level_counts.values, color=colors, edgecolor='black')
    ax.set_ylabel('Number of Districts', fontsize=12)
    ax.set_title('IPCC Risk Level Distribution', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    count += save_fig(fig, folder, 'F01_ipcc_levels.png')
    
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    
    ipcc_sorted = ipcc.sort_values('hotspot_score', ascending=True)
    colors = [RISK_COLORS.get(level, '#888888') for level in ipcc_sorted['ipcc_level']]
    
    ax.barh(ipcc_sorted['district'], ipcc_sorted['hotspot_score'], color=colors, edgecolor='black')
    ax.set_xlabel('Risk Score', fontsize=12)
    ax.set_title('IPCC Risk Score by District', fontsize=14, fontweight='bold')
    
    count += save_fig(fig, folder, 'F02_risk_by_district.png')
    
    print(f"  Section F complete: {count} figures")
    return count


def plot_section_G(data):
    """Climate projections visualizations."""
    print("\n" + "=" * 70)
    print("SECTION G: CLIMATE PROJECTIONS")
    print("=" * 70)
    
    folder = FOLDERS['G']
    count = 0
    
    if 'projections' not in data:
        print("  ✗ Required data not available")
        return 0
    
    proj = data['projections']
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    for district in proj['district'].unique():
        d_proj = proj[proj['district'] == district]
        ax.plot(d_proj['year'], d_proj['projected_anomaly'], marker='o', linewidth=1.5, 
               alpha=0.7, label=district)
    
    ax.axhline(y=1.5, color='orange', linestyle='--', linewidth=2, label='1.5°C Target')
    ax.axhline(y=2.0, color='red', linestyle='--', linewidth=2, label='2.0°C Limit')
    
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Projected Temperature Anomaly (°C)', fontsize=12)
    ax.set_title('Climate Projections for All Districts (2030-2080)', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    count += save_fig(fig, folder, 'G01_all_projections.png')
    
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    
    if 'ranking' in data:
        top5 = data['ranking'].head(5)['district'].tolist()
        colors = plt.cm.Reds(np.linspace(0.4, 0.9, 5))
        
        for i, district in enumerate(top5):
            d_proj = proj[proj['district'] == district]
            ax.plot(d_proj['year'], d_proj['projected_anomaly'], marker='o', linewidth=2,
                   color=colors[i], label=district)
        
        ax.axhline(y=1.5, color='orange', linestyle='--', linewidth=2)
        ax.axhline(y=2.0, color='red', linestyle='--', linewidth=2)
        
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Projected Anomaly (°C)', fontsize=12)
        ax.set_title('Climate Projections: Top 5 Hotspots', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        count += save_fig(fig, folder, 'G02_top5_projections.png')
    
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    
    stats = proj.groupby('year').agg({
        'projected_anomaly': ['mean', 'std', 'min', 'max']
    }).reset_index()
    stats.columns = ['year', 'mean', 'std', 'min', 'max']
    
    ax.fill_between(stats['year'], stats['min'], stats['max'], alpha=0.2, color='red', label='Range')
    ax.fill_between(stats['year'], stats['mean'] - stats['std'], stats['mean'] + stats['std'],
                   alpha=0.4, color='blue', label='±1 Std')
    ax.plot(stats['year'], stats['mean'], 'b-o', linewidth=2, label='Mean')
    
    ax.axhline(y=1.5, color='orange', linestyle='--', linewidth=2)
    ax.axhline(y=2.0, color='red', linestyle='--', linewidth=2)
    
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Projected Anomaly (°C)', fontsize=12)
    ax.set_title('Projection Uncertainty', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    count += save_fig(fig, folder, 'G03_uncertainty.png')
    
    print(f"  Section G complete: {count} figures")
    return count


def plot_section_H(data):
    """Policy recommendations visualizations."""
    print("\n" + "=" * 70)
    print("SECTION H: POLICY RECOMMENDATIONS")
    print("=" * 70)
    
    folder = FOLDERS['H']
    count = 0
    
    if 'ranking' not in data:
        print("  ✗ Required data not available")
        return 0
    
    ranking = data['ranking']
    
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    
    total_budget = 100
    ranking['budget_share'] = ranking['hotspot_score'] / ranking['hotspot_score'].sum() * total_budget
    ranking_sorted = ranking.sort_values('budget_share', ascending=True)
    
    colors = [RISK_COLORS.get(level, '#888888') for level in ranking_sorted['priority_level']]
    ax.barh(ranking_sorted['district'], ranking_sorted['budget_share'], color=colors, edgecolor='black')
    
    ax.set_xlabel('Recommended Budget (Billion PKR)', fontsize=12)
    ax.set_title('Climate Adaptation Budget Priorities', fontsize=14, fontweight='bold')
    
    count += save_fig(fig, folder, 'H01_budget_allocation.png')
    
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    
    priority_order = ['Critical', 'High', 'Medium', 'Low']
    priority_counts = ranking['priority_level'].value_counts()
    priority_counts = priority_counts.reindex(priority_order).fillna(0)
    
    colors = [RISK_COLORS.get(p, '#888888') for p in priority_order]
    ax.bar(priority_order, priority_counts.values, color=colors, edgecolor='black')
    ax.set_ylabel('Number of Districts', fontsize=12)
    ax.set_title('Priority Level Distribution', fontsize=14, fontweight='bold')
    
    for i, v in enumerate(priority_counts.values):
        ax.text(i, v + 0.2, str(int(v)), ha='center', fontsize=12, fontweight='bold')
    
    count += save_fig(fig, folder, 'H02_priority_distribution.png')
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 6))
    
    top5 = ranking.head(5)
    
    for ax, (_, row) in zip(axes, top5.iterrows()):
        ax.axis('off')
        color = RISK_COLORS.get(row['priority_level'], '#888888')
        
        rect = plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                             facecolor=color, alpha=0.2, edgecolor=color, linewidth=3)
        ax.add_patch(rect)
        
        text = f"""
{row['district']}
───────────
Priority: {row['priority_level']}
Score: {row['hotspot_score']:.2f}
R²: {row['model_r2']:.2f}
Rank: #{int(row['priority_rank'])}
        """
        
        ax.text(0.5, 0.5, text, transform=ax.transAxes, fontsize=10,
               verticalalignment='center', horizontalalignment='center',
               fontfamily='monospace')
    
    fig.suptitle('Top 5 Priority Districts', fontsize=14, fontweight='bold')
    plt.tight_layout()
    count += save_fig(fig, folder, 'H03_action_cards.png')
    
    print(f"  Section H complete: {count} figures")
    return count


def plot_section_I(data):
    """Summary visualizations."""
    print("\n" + "=" * 70)
    print("SECTION I: SUMMARY")
    print("=" * 70)
    
    folder = FOLDERS['I']
    count = 0
    
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.axis('off')
    
    lodo = data.get('lodo', pd.DataFrame())
    ranking = data.get('ranking', pd.DataFrame())
    
    lodo_mean = lodo['r2'].mean() if len(lodo) > 0 else 0
    lodo_median = lodo['r2'].median() if len(lodo) > 0 else 0
    
    text = f"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                    SINDH CLIMATE HOTSPOT DETECTION - KEY RESULTS                ║
╠══════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                      ║
║  📊 MODEL PERFORMANCE (LODO-CV):                                                     ║
║     Mean R²: {lodo_mean:.4f}                                                              ║
║     Median R²: {lodo_median:.4f}                                                            ║
║     Districts R² > 0.90: {(lodo['r2'] > 0.9).sum() if len(lodo) > 0 else 0}/22                                                          ║
║     Districts R² > 0.80: {(lodo['r2'] > 0.8).sum() if len(lodo) > 0 else 0}/22                                                          ║
║                                                                                      ║
║  🔥 TOP 5 CLIMATE HOTSPOTS:                                                          ║
"""
    
    if len(ranking) > 0:
        for _, row in ranking.head(5).iterrows():
            text += f"║     {int(row['priority_rank'])}. {row['district']:20s} (Score: {row['hotspot_score']:.3f}, {row['priority_level']})              ║\n"
    
    text += """║                                                                                      ║
║  ✅ METHODOLOGY: Leave-One-District-Out Cross-Validation                             ║
║  📈 DATA: 44 years (1981-2024), 22 districts, 968 observations                       ║
║                                                                                      ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
    """
    
    ax.text(0.5, 0.5, text, transform=ax.transAxes, fontsize=10,
           verticalalignment='center', horizontalalignment='center',
           fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow'))
    
    count += save_fig(fig, folder, 'I01_key_results.png')
    
    if len(ranking) > 0 and len(lodo) > 0:
        fig, ax = plt.subplots(figsize=(14, 12))
        ax.axis('off')
        
        table_data = ranking.merge(lodo[['district', 'r2', 'rmse']], on='district')
        table_data = table_data[['priority_rank', 'district', 'r2', 'rmse', 'hotspot_score', 'priority_level']]
        table_data = table_data.head(15)
        
        table = ax.table(
            cellText=table_data.round(3).values,
            colLabels=['Rank', 'District', 'R²', 'RMSE', 'Score', 'Priority'],
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        ax.set_title('Publication Table: Top 15 Districts', fontsize=14, fontweight='bold', pad=20)
        
        count += save_fig(fig, folder, 'I02_publication_table.png')
    
    print(f"  Section I complete: {count} figures")
    return count

def main():
    print("\n" + "=" * 70)
    print("VISUALIZATION ...")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    data = load_v4_data()
    
    total = 0
    total += plot_section_A(data)
    total += plot_section_B(data)
    total += plot_section_C(data)
    total += plot_section_D(data)
    total += plot_section_E(data)
    total += plot_section_F(data)
    total += plot_section_G(data)
    total += plot_section_H(data)
    total += plot_section_I(data)
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ VISUALIZATION COMPLETE")
    print("=" * 70)
    print(f"\n  Total figures generated: {total}")
    
    for key, folder in FOLDERS.items():
        n_files = len(list((FIGURES_DIR / folder).glob('*.png')))
        if n_files > 0:
            print(f"    {folder}: {n_files} figures")
    
    print(f"\n  Output directory: {FIGURES_DIR}")
    print("=" * 70)
    
    return total

if __name__ == "__main__":
    main()
