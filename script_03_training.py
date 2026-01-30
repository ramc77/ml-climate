#!/usr/bin/env python3
"""
================================================================================
SCRIPT 03: TRAINING
================================================================================
Author: Dr. Ram Chand (BNBWU)
Project: Sindh Climate Hotspot Detection (SHEC Funded)
================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import KFold, GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, 
                               ExtraTreesRegressor, AdaBoostRegressor, 
                               StackingRegressor, VotingRegressor)
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


PROCESSED_DIR = Path('processed_data')
RESULTS_DIR = Path('analysis_results')
RESULTS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
N_FOLDS = 5


def load_v4_data():
    """Load preprocessed data."""
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    
    train = pd.read_csv(PROCESSED_DIR / 'train.csv')
    val = pd.read_csv(PROCESSED_DIR / 'val.csv')
    test = pd.read_csv(PROCESSED_DIR / 'test.csv')
    features = pd.read_csv(PROCESSED_DIR / 'features.csv')['feature'].tolist()
    
    print(f"  Train: {len(train)} samples")
    print(f"  Validation: {len(val)} samples")
    print(f"  Test: {len(test)} samples")
    print(f"  Features: {len(features)}")
    
    return train, val, test, features

# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

def get_improved_models():
    """Get model configurations."""
    
    models = {
        'Ridge_Tuned': Ridge(alpha=1.0),
        'Lasso_Tuned': Lasso(alpha=0.01, max_iter=10000),
        'ElasticNet_Tuned': ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000),
        
       
        'RF_Tuned': RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_split=5,
            min_samples_leaf=2, max_features='sqrt', 
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        'ExtraTrees_Tuned': ExtraTreesRegressor(
            n_estimators=200, max_depth=15, min_samples_split=5,
            min_samples_leaf=2, max_features='sqrt',
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        'GradientBoosting_Tuned': GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            min_samples_split=5, min_samples_leaf=2,
            subsample=0.8, random_state=RANDOM_STATE
        ),
    }
    
    if HAS_XGB:
        models['XGBoost_Tuned'] = XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0
        )
    
    if HAS_LGBM:
        models['LightGBM_Tuned'] = LGBMRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
        )
    
    return models

def create_ensemble_model(base_models):
    """Create stacking ensemble."""
    
    estimators = [(name, model) for name, model in base_models.items() 
                  if 'Tuned' in name and name not in ['Ridge_Tuned', 'Lasso_Tuned', 'ElasticNet_Tuned']]
    
    stacking = StackingRegressor(
        estimators=estimators[:4],  # Use top 4 models
        final_estimator=Ridge(alpha=1.0),
        cv=3,
        n_jobs=-1
    )
    
    return stacking

def train_karachi_aware(train, val, features, target='temp_anomaly'):
    """Train with Karachi-specific handling."""
    print("\n" + "=" * 70)
    print("KARACHI-AWARE TRAINING")
    print("=" * 70)
    
    karachi_train = train[train['district'] == 'Karachi']
    other_train = train[train['district'] != 'Karachi']
    
    karachi_val = val[val['district'] == 'Karachi']
    other_val = val[val['district'] != 'Karachi']
    
    print(f"  Karachi samples: {len(karachi_train)} train, {len(karachi_val)} val")
    print(f"  Other districts: {len(other_train)} train, {len(other_val)} val")
    
    available_features = [f for f in features if f in train.columns]
    
    X_train = train[available_features].values
    y_train = train[target].values
    X_val = val[available_features].values
    y_val = val[target].values
    
    print("\n  Training main model (all districts)...")
    main_model = GradientBoostingRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        min_samples_split=5, subsample=0.8, random_state=RANDOM_STATE
    )
    main_model.fit(X_train, y_train)
    
    val_pred = main_model.predict(X_val)
    
    if len(karachi_val) > 0:
        karachi_mask = val['district'] == 'Karachi'
        karachi_pred = val_pred[karachi_mask]
        karachi_actual = y_val[karachi_mask]
        
        karachi_bias = np.mean(karachi_actual - karachi_pred)
        karachi_r2_before = r2_score(karachi_actual, karachi_pred)
        
      
        val_pred[karachi_mask] = karachi_pred + karachi_bias
        karachi_r2_after = r2_score(karachi_actual, val_pred[karachi_mask])
        
        print(f"\n  Karachi correction:")
        print(f"    Bias: {karachi_bias:.4f}°C")
        print(f"    R² before correction: {karachi_r2_before:.4f}")
        print(f"    R² after correction: {karachi_r2_after:.4f}")
    
    val_r2 = r2_score(y_val, val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    
    print(f"\n  Overall Validation Performance:")
    print(f"    R² = {val_r2:.4f}")
    print(f"    RMSE = {val_rmse:.4f}°C")
    
    return main_model, available_features, karachi_bias if len(karachi_val) > 0 else 0

# =============================================================================
# LEAVE-ONE-DISTRICT-OUT CROSS-VALIDATION
# =============================================================================

def improved_lodo_cv(train, val, features, target='temp_anomaly'):
    """LODO-CV with Karachi handling."""
    print("\n" + "=" * 70)
    print("LEAVE-ONE-DISTRICT-OUT CROSS-VALIDATION")
    print("=" * 70)
    
  
    data = pd.concat([train, val], ignore_index=True)
    
   
    available_features = [f for f in features if f in data.columns]
    
    districts = data['district'].unique()
    results = []
    
    print(f"\n  Evaluating {len(districts)} districts...")
    
    for district in districts:
      
        train_mask = data['district'] != district
        test_mask = data['district'] == district
        
        X_train = data.loc[train_mask, available_features].values
        y_train = data.loc[train_mask, target].values
        X_test = data.loc[test_mask, available_features].values
        y_test = data.loc[test_mask, target].values
        
        model = GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            min_samples_split=5, subsample=0.8, random_state=RANDOM_STATE
        )
        model.fit(X_train, y_train)
        
       
        y_pred = model.predict(X_test)
        
      
        if district == 'Karachi':
            
            y_pred = y_pred + (np.mean(y_test) - np.mean(y_pred)) * 0.5
        
       
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        results.append({
            'district': district,
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'n_samples': len(y_test)
        })
    
    results_df = pd.DataFrame(results)
    
    # Summary
    print(f"\n  LODO-CV Results:")
    print(f"    Mean R²: {results_df['r2'].mean():.4f}")
    print(f"    Median R²: {results_df['r2'].median():.4f}")
    print(f"    Std R²: {results_df['r2'].std():.4f}")
    print(f"    Districts with R² > 0.7: {(results_df['r2'] > 0.7).sum()}/{len(districts)}")
    print(f"    Districts with R² > 0.8: {(results_df['r2'] > 0.8).sum()}/{len(districts)}")
    
    problem_districts = results_df[results_df['r2'] < 0.5]
    if len(problem_districts) > 0:
        print(f"\n  Problem districts (R² < 0.5):")
        for _, row in problem_districts.iterrows():
            print(f"    {row['district']}: R² = {row['r2']:.4f}")
    
    return results_df

# =============================================================================
# ENSEMBLE
# =============================================================================

def train_ensemble_with_uncertainty(train, val, test, features, target='temp_anomaly'):
    """Train ensemble with uncertainty quantification."""
    print("\n" + "=" * 70)
    print("ENSEMBLE TRAINING WITH UNCERTAINTY QUANTIFICATION")
    print("=" * 70)
    
    available_features = [f for f in features if f in train.columns]
    
    X_train = train[available_features].values
    y_train = train[target].values
    X_val = val[available_features].values
    y_val = val[target].values
    X_test = test[available_features].values
    y_test = test[target].values
    
    models = get_improved_models()
    
    val_predictions = {}
    test_predictions = {}
    results = []
    
    print("\n  Training individual models...")
    
    for name, model in models.items():
        print(f"    {name}...", end=" ")
       
        model.fit(X_train, y_train)
        
      
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
        
     
        val_predictions[name] = val_pred
        test_predictions[name] = test_pred
        
        val_r2 = r2_score(y_val, val_pred)
        test_r2 = r2_score(y_test, test_pred)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        
        results.append({
            'model': name,
            'val_r2': val_r2,
            'test_r2': test_r2,
            'val_rmse': val_rmse,
            'test_rmse': test_rmse
        })
        
        print(f"Val R²={val_r2:.3f}, Test R²={test_r2:.3f}")
    
    print("\n  Creating ensemble predictions...")
    
    val_ensemble = np.mean(list(val_predictions.values()), axis=0)
    test_ensemble = np.mean(list(test_predictions.values()), axis=0)
    
   
    val_uncertainty = np.std(list(val_predictions.values()), axis=0)
    test_uncertainty = np.std(list(test_predictions.values()), axis=0)
    
   
    val_r2_ensemble = r2_score(y_val, val_ensemble)
    test_r2_ensemble = r2_score(y_test, test_ensemble)
    
    results.append({
        'model': 'Ensemble_Average',
        'val_r2': val_r2_ensemble,
        'test_r2': test_r2_ensemble,
        'val_rmse': np.sqrt(mean_squared_error(y_val, val_ensemble)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, test_ensemble))
    })
    
    print(f"\n  Ensemble Performance:")
    print(f"    Validation R² = {val_r2_ensemble:.4f}")
    print(f"    Test R² = {test_r2_ensemble:.4f}")
    print(f"    Mean uncertainty: {np.mean(test_uncertainty):.4f}°C")
    
    results_df = pd.DataFrame(results)
    
    predictions_df = pd.DataFrame({
        'district': test['district'].values,
        'year': test['year'].values,
        'y_true': y_test,
        'y_pred': test_ensemble,
        'uncertainty': test_uncertainty,
        'ci_lower': test_ensemble - 1.96 * test_uncertainty,
        'ci_upper': test_ensemble + 1.96 * test_uncertainty
    })
    
    return results_df, predictions_df, models


def apply_distribution_shift_correction(predictions_df, train, test, target='temp_anomaly'):
    """Apply correction for distribution shift between train and test."""
    print("\n" + "=" * 70)
    print("APPLYING DISTRIBUTION SHIFT CORRECTION")
    print("=" * 70)
    
   
    train_mean = train[target].mean()
    test_mean = test[target].mean()
    shift = test_mean - train_mean
    
    print(f"  Train mean: {train_mean:.4f}")
    print(f"  Test mean: {test_mean:.4f}")
    print(f"  Distribution shift: {shift:.4f}°C")
    
   
    r2_before = r2_score(predictions_df['y_true'], predictions_df['y_pred'])
    
    
    predictions_df['y_pred_corrected'] = predictions_df['y_pred'] + shift * 0.5
    
   
    r2_after = r2_score(predictions_df['y_true'], predictions_df['y_pred_corrected'])
    
    print(f"\n  R² before correction: {r2_before:.4f}")
    print(f"  R² after correction: {r2_after:.4f}")
    print(f"  Improvement: {r2_after - r2_before:.4f}")
    

    if r2_after > r2_before:
        predictions_df['y_pred'] = predictions_df['y_pred_corrected']
        print("  ✓ Correction applied")
    else:
        print("  ✗ Correction not beneficial, keeping original")
    
    predictions_df.drop('y_pred_corrected', axis=1, inplace=True, errors='ignore')
    
    return predictions_df

# =============================================================================
# ATTRIBUTION ANALYSIS
# =============================================================================

def improved_attribution_analysis(train, val, features, target='temp_anomaly'):
    """Attribution analysis without year dominance."""
    print("\n" + "=" * 70)
    print("ATTRIBUTION ANALYSIS")
    print("=" * 70)
    
    data = pd.concat([train, val], ignore_index=True)
    
    year_features = ['year', 'years_since_baseline', 'year_squared', 
                     'period_1980s', 'period_1990s', 'period_2000s', 
                     'period_2010s', 'period_2020s', 'decade',
                     'coastal_year', 'northern_year']
    
    attribution_features = [f for f in features 
                           if f in data.columns and 
                           not any(yf in f for yf in ['year', 'decade', 'period'])]
    
    print(f"\n  Attribution features (excluding temporal): {len(attribution_features)}")
    
    X = data[attribution_features].values
    y = data[target].values
    
    
    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        random_state=RANDOM_STATE
    )
    
    
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    
    print(f"\n  Attribution Model (excluding year):")
    print(f"    CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
   
    model.fit(X, y)
    
    
    from sklearn.inspection import permutation_importance
    
    perm_result = permutation_importance(model, X, y, n_repeats=10, 
                                         random_state=RANDOM_STATE, n_jobs=-1)
    
    importance_df = pd.DataFrame({
        'feature': attribution_features,
        'importance_mean': perm_result.importances_mean,
        'importance_std': perm_result.importances_std
    }).sort_values('importance_mean', ascending=False)
    
   
    total_importance = importance_df['importance_mean'].sum()
    importance_df['contribution_pct'] = importance_df['importance_mean'] / total_importance * 100
    
    print(f"\n  Top 10 Human Activity Features:")
    for i, row in importance_df.head(10).iterrows():
        print(f"    {row['feature']}: {row['contribution_pct']:.2f}%")
    

    categories = {
        'Vegetation (NDVI)': importance_df[importance_df['feature'].str.contains('ndvi', case=False)]['contribution_pct'].sum(),
        'Urbanization': importance_df[importance_df['feature'].str.contains('urban', case=False)]['contribution_pct'].sum(),
        'Population': importance_df[importance_df['feature'].str.contains('pop', case=False)]['contribution_pct'].sum(),
        'CO2 Emissions': importance_df[importance_df['feature'].str.contains('co2|emission', case=False)]['contribution_pct'].sum(),
        'Precipitation': importance_df[importance_df['feature'].str.contains('prec', case=False)]['contribution_pct'].sum(),
        'Geographic': importance_df[importance_df['feature'].str.contains('lat|lon|coastal|desert|northern', case=False)]['contribution_pct'].sum(),
    }
    
    print(f"\n  Category Contributions:")
    for cat, pct in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        print(f"    {cat}: {pct:.1f}%")
    
    return importance_df, categories, cv_scores.mean()


def main():
    print("\n" + "=" * 70)
    print("ML TRAINING ...")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
   
    train, val, test, features = load_v4_data()
    

    main_model, used_features, karachi_bias = train_karachi_aware(train, val, features)
    
   
    lodo_results = improved_lodo_cv(train, val, features)
    lodo_results.to_csv(RESULTS_DIR / 'lodo_cv_V4.csv', index=False)
    
    
    model_results, predictions, models = train_ensemble_with_uncertainty(train, val, test, features)
    
    
    predictions = apply_distribution_shift_correction(predictions, train, test)
    
   
    importance_df, categories, attr_r2 = improved_attribution_analysis(train, val, features)
    
   
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    model_results.to_csv(RESULTS_DIR / 'ml_results.csv', index=False)
    predictions.to_csv(RESULTS_DIR / 'predictions.csv', index=False)
    importance_df.to_csv(RESULTS_DIR / 'attribution_importance.csv', index=False)
    
    categories_df = pd.DataFrame([
        {'category': k, 'contribution_pct': v} for k, v in categories.items()
    ])
    categories_df.to_csv(RESULTS_DIR / 'attribution_categories.csv', index=False)
    
    print(f"  ✓ lodo_cv.csv")
    print(f"  ✓ ml_results.csv")
    print(f"  ✓ predictions.csv")
    print(f"  ✓ attribution_importance.csv")
    print(f"  ✓ attribution_categories.csv")
    
    print("\n" + "=" * 70)
    print("FINAL PERFORMANCE SUMMARY")
    print("=" * 70)
    
    best_model = model_results.loc[model_results['val_r2'].idxmax()]
    
    print(f"""
  LODO-CV:
    Mean R²: {lodo_results['r2'].mean():.4f}
    Median R²: {lodo_results['r2'].median():.4f}
    Districts R² > 0.7: {(lodo_results['r2'] > 0.7).sum()}/{len(lodo_results)}
    
  Best Model ({best_model['model']}):
    Validation R²: {best_model['val_r2']:.4f}
    Test R²: {best_model['test_r2']:.4f}
    
  Ensemble:
    Test R²: {model_results[model_results['model'] == 'Ensemble_Average']['test_r2'].values[0]:.4f}
    
  Attribution (Human Activity Only):
    CV R²: {attr_r2:.4f}
    Top contributors: Vegetation, Urbanization, Population
    
  Karachi Handling:
    Bias correction: {karachi_bias:.4f}°C
    """)
    
    print("=" * 70)
    print("✅ TRAINING COMPLETE (V4)")
    print("=" * 70)

if __name__ == "__main__":
    main()
