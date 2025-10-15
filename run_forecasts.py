#!/usr/bin/env python
"""
Run rolling forecasts for all models - sequential execution with progress saving.
This script runs each model separately to avoid timeouts and allow progress tracking.
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Suppress Hugging Face progress bars
import os
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)

# Paths
project_root = Path(__file__).parent
data_dir = project_root / 'data'
results_dir = project_root / 'results' / 'forecasts'
results_dir.mkdir(parents=True, exist_ok=True)

# Load data
print("Loading data...")
ts = pd.read_pickle(data_dir / 'flu_daily_clean.pkl')
print(f"Loaded {len(ts)} observations from {ts.index.min()} to {ts.index.max()}")

# Configuration - OPTIMIZED for API efficiency
HORIZONS = [7, 28]  # Short-term (weekly) vs long-term (monthly) forecasts
ORIGINS = pd.date_range('2024-07-08', '2025-05-26', freq='2W-MON')  # Bi-weekly
MIN_TRAIN = 730

print(f"\nConfiguration (OPTIMIZED):")
print(f"  Horizons: {HORIZONS} (7-day and 28-day ahead)")
print(f"  Origins: {len(ORIGINS)} (bi-weekly Mondays)")
print(f"  Total forecasts per model: {len(ORIGINS) * len(HORIZONS)}")
print(f"  ✅ API-friendly: 48 calls per model (24 time points × 2 horizons)")

# ============================================================================
# Helper Functions
# ============================================================================

def build_fourier_terms(dates, period=365, order=2):
    """Build Fourier terms for seasonality."""
    fourier = pd.DataFrame(index=dates)
    for k in range(1, order + 1):
        fourier[f'sin{k}'] = np.sin(2 * np.pi * k * np.arange(len(dates)) / period)
        fourier[f'cos{k}'] = np.cos(2 * np.pi * k * np.arange(len(dates)) / period)
    return fourier

def build_lag_features(series, lags=[1, 2, 3, 7, 14]):
    """Build lag features."""
    lag_df = pd.DataFrame(index=series.index)
    for lag in lags:
        lag_df[f'lag_{lag}'] = series.shift(lag)
    return lag_df

# ============================================================================
# Model 1: SARIMA + Fourier
# ============================================================================

def forecast_sarima_fourier(train_series, horizon, order=(1, 0, 1), fourier_order=2, period=365):
    """SARIMA + Fourier forecast."""
    import statsmodels.api as sm

    fourier_train = build_fourier_terms(train_series.index, period=period, order=fourier_order)
    forecast_start = train_series.index[-1] + pd.Timedelta(days=1)
    forecast_dates = pd.date_range(forecast_start, periods=horizon, freq='D')
    fourier_test = build_fourier_terms(forecast_dates, period=period, order=fourier_order)

    model = sm.tsa.SARIMAX(
        train_series,
        order=order,
        seasonal_order=(0, 0, 0, 0),
        trend='c',
        exog=fourier_train
    )

    try:
        fit = model.fit(disp=False, maxiter=200)
        forecast = fit.forecast(steps=horizon, exog=fourier_test)
        point = forecast.iloc[-1]
        forecast_obj = fit.get_forecast(steps=horizon, exog=fourier_test)
        pred_int = forecast_obj.conf_int(alpha=0.2)

        return {
            'q0.1': float(pred_int.iloc[-1, 0]),
            'q0.5': float(point),
            'q0.9': float(pred_int.iloc[-1, 1])
        }
    except:
        naive = float(train_series.iloc[-1])
        return {'q0.1': naive * 0.8, 'q0.5': naive, 'q0.9': naive * 1.2}

def run_sarima():
    """Run SARIMA forecasts."""
    print("\n" + "="*60)
    print("Running SARIMA + Fourier")
    print("="*60)

    results = []
    pbar = tqdm(total=len(ORIGINS) * len(HORIZONS), desc="SARIMA")

    for origin in ORIGINS:
        train = ts[ts.index < origin]
        if len(train) < MIN_TRAIN:
            pbar.update(len(HORIZONS))
            continue

        for horizon in HORIZONS:
            target_date = origin + pd.Timedelta(days=horizon - 1)
            if target_date not in ts.index:
                pbar.update(1)
                continue

            actual = ts.loc[target_date]

            try:
                pred = forecast_sarima_fourier(train, horizon)
                results.append({
                    'date': target_date,
                    'origin': origin,
                    'horizon': horizon,
                    'model': 'SARIMA_Fourier',
                    'q0.1': pred['q0.1'],
                    'q0.5': pred['q0.5'],
                    'q0.9': pred['q0.9'],
                    'actual': actual
                })
            except:
                pass

            pbar.update(1)

    pbar.close()

    df = pd.DataFrame(results)
    output_path = results_dir / 'sarima_fourier.parquet'
    df.to_parquet(output_path, index=False)
    print(f"✅ Saved {len(df)} forecasts to {output_path}")
    return df

# ============================================================================
# Model 2: LightGBM
# ============================================================================

def forecast_lightgbm(train_series, horizon, lags=[1, 2, 3, 7, 14], fourier_order=2, period=365):
    """LightGBM forecast."""
    import lightgbm as lgb

    lag_df = build_lag_features(train_series, lags=lags)
    fourier_df = build_fourier_terms(train_series.index, period=period, order=fourier_order)
    X_train = pd.concat([lag_df, fourier_df], axis=1).dropna()
    y_train = train_series.loc[X_train.index]

    predictions = {}
    for q in [0.1, 0.5, 0.9]:
        model = lgb.LGBMRegressor(
            objective='quantile',
            alpha=q,
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            verbose=-1
        )
        model.fit(X_train, y_train)

        current_series = train_series.copy()
        for step in range(horizon):
            lag_feats = build_lag_features(current_series, lags=lags).iloc[-1:]
            next_date = current_series.index[-1] + pd.Timedelta(days=1)
            fourier_feats = build_fourier_terms(pd.DatetimeIndex([next_date]), period=period, order=fourier_order)
            X_next = pd.concat([lag_feats, fourier_feats], axis=1)
            pred = model.predict(X_next)[0]
            current_series = pd.concat([current_series, pd.Series([pred], index=[next_date])])

        predictions[f'q{q}'] = float(pred)

    # Enforce quantile monotonicity: q0.1 ≤ q0.5 ≤ q0.9
    # This fixes the quantile crossing issue in LightGBM quantile regression
    predictions['q0.1'] = min(predictions['q0.1'], predictions['q0.5'])
    predictions['q0.9'] = max(predictions['q0.5'], predictions['q0.9'])

    return predictions

def run_lightgbm():
    """Run LightGBM forecasts."""
    print("\n" + "="*60)
    print("Running LightGBM")
    print("="*60)

    results = []
    pbar = tqdm(total=len(ORIGINS) * len(HORIZONS), desc="LightGBM")

    for origin in ORIGINS:
        train = ts[ts.index < origin]
        if len(train) < MIN_TRAIN:
            pbar.update(len(HORIZONS))
            continue

        for horizon in HORIZONS:
            target_date = origin + pd.Timedelta(days=horizon - 1)
            if target_date not in ts.index:
                pbar.update(1)
                continue

            actual = ts.loc[target_date]

            try:
                pred = forecast_lightgbm(train, horizon)
                results.append({
                    'date': target_date,
                    'origin': origin,
                    'horizon': horizon,
                    'model': 'LightGBM',
                    'q0.1': pred['q0.1'],
                    'q0.5': pred['q0.5'],
                    'q0.9': pred['q0.9'],
                    'actual': actual
                })
            except:
                pass

            pbar.update(1)

    pbar.close()

    df = pd.DataFrame(results)
    output_path = results_dir / 'lightgbm.parquet'
    df.to_parquet(output_path, index=False)
    print(f"✅ Saved {len(df)} forecasts to {output_path}")
    return df

# ============================================================================
# Model 3: Chronos-Tiny
# ============================================================================

def forecast_chronos(train_series, horizon, pipeline, num_samples=100):
    """Chronos forecast."""
    context = torch.tensor(train_series.values, dtype=torch.float32)
    forecast_samples = pipeline.predict(
        context=context.unsqueeze(0),
        prediction_length=horizon,
        num_samples=num_samples
    )
    final_preds = forecast_samples[0, :, -1].numpy()

    return {
        'q0.1': float(np.quantile(final_preds, 0.1)),
        'q0.5': float(np.quantile(final_preds, 0.5)),
        'q0.9': float(np.quantile(final_preds, 0.9))
    }

def run_chronos():
    """Run Chronos forecasts."""
    print("\n" + "="*60)
    print("Running Chronos-Tiny")
    print("="*60)

    from chronos import ChronosPipeline

    print("Loading Chronos model...")
    chronos_pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-tiny",
        device_map="cpu",
        torch_dtype=torch.bfloat16,
    )
    print("✅ Model loaded")

    results = []
    pbar = tqdm(total=len(ORIGINS) * len(HORIZONS), desc="Chronos")

    for origin in ORIGINS:
        train = ts[ts.index < origin]
        if len(train) < MIN_TRAIN:
            pbar.update(len(HORIZONS))
            continue

        for horizon in HORIZONS:
            target_date = origin + pd.Timedelta(days=horizon - 1)
            if target_date not in ts.index:
                pbar.update(1)
                continue

            actual = ts.loc[target_date]

            try:
                pred = forecast_chronos(train, horizon, chronos_pipeline)
                results.append({
                    'date': target_date,
                    'origin': origin,
                    'horizon': horizon,
                    'model': 'Chronos_Tiny',
                    'q0.1': pred['q0.1'],
                    'q0.5': pred['q0.5'],
                    'q0.9': pred['q0.9'],
                    'actual': actual
                })
            except:
                pass

            pbar.update(1)

    pbar.close()

    df = pd.DataFrame(results)
    output_path = results_dir / 'chronos.parquet'
    df.to_parquet(output_path, index=False)
    print(f"✅ Saved {len(df)} forecasts to {output_path}")
    return df

# ============================================================================
# Model 4: TabPFN-TS (Now optimized to 48 calls!)
# ============================================================================

def forecast_tabpfn(train_series, horizon, item_id='flu_positivity'):
    """TabPFN-TS forecast."""
    from autogluon.timeseries import TimeSeriesDataFrame
    from tabpfn_time_series import TabPFNTimeSeriesPredictor, TabPFNMode

    df_prep = train_series.reset_index()
    df_prep.columns = ['timestamp', 'target']
    df_prep['item_id'] = item_id
    df_prep['day_of_year'] = df_prep['timestamp'].dt.dayofyear
    df_prep['month'] = df_prep['timestamp'].dt.month

    train_tsdf = TimeSeriesDataFrame.from_data_frame(
        df_prep[['item_id', 'timestamp', 'target', 'day_of_year', 'month']],
        id_column='item_id',
        timestamp_column='timestamp'
    )

    forecast_start = train_series.index[-1] + pd.Timedelta(days=1)
    test_dates = pd.date_range(forecast_start, periods=horizon, freq='D')

    test_df = pd.DataFrame({
        'timestamp': test_dates,
        'item_id': item_id,
        'target': np.nan,
        'day_of_year': test_dates.dayofyear,
        'month': test_dates.month
    })

    test_tsdf = TimeSeriesDataFrame.from_data_frame(
        test_df,
        id_column='item_id',
        timestamp_column='timestamp'
    )

    predictor = TabPFNTimeSeriesPredictor(tabpfn_mode=TabPFNMode.CLIENT)
    pred = predictor.predict(train_tsdf, test_tsdf)
    pred_slice = pred.loc[item_id]

    return {
        'q0.1': float(pred_slice[0.1].iloc[-1]),
        'q0.5': float(pred_slice[0.5].iloc[-1]),
        'q0.9': float(pred_slice[0.9].iloc[-1])
    }

def run_tabpfn():
    """Run TabPFN forecasts (now optimized to 48 API calls)."""
    print("\n" + "="*60)
    print("Running TabPFN-TS (48 API calls)")
    print("="*60)
    print("⚠️  This requires TabPFN API access.")
    print("   Set SKIP_TABPFN=1 environment variable to skip.")

    # Allow skipping via environment variable
    if os.getenv('SKIP_TABPFN', '0') == '1':
        print("   Skipping TabPFN (SKIP_TABPFN=1)")
        return None

    # Check for API key
    from dotenv import load_dotenv
    import tabpfn_client

    env_file = project_root / ".env"
    if not env_file.exists():
        print("   ⚠️  .env file not found, skipping TabPFN")
        return None

    load_dotenv(env_file)
    api_key = os.getenv("PRIORLABS_API_KEY")
    if not api_key:
        print("   ⚠️  PRIORLABS_API_KEY not found in .env, skipping TabPFN")
        return None

    tabpfn_client.set_access_token(api_key)
    print("   ✅ API key loaded")

    results = []
    pbar = tqdm(total=len(ORIGINS) * len(HORIZONS), desc="TabPFN")

    for origin in ORIGINS:
        train = ts[ts.index < origin]
        if len(train) < MIN_TRAIN:
            pbar.update(len(HORIZONS))
            continue

        for horizon in HORIZONS:
            target_date = origin + pd.Timedelta(days=horizon - 1)
            if target_date not in ts.index:
                pbar.update(1)
                continue

            actual = ts.loc[target_date]

            try:
                pred = forecast_tabpfn(train, horizon)
                results.append({
                    'date': target_date,
                    'origin': origin,
                    'horizon': horizon,
                    'model': 'TabPFN_TS',
                    'q0.1': pred['q0.1'],
                    'q0.5': pred['q0.5'],
                    'q0.9': pred['q0.9'],
                    'actual': actual
                })
            except Exception as e:
                print(f"\n   ⚠️  Failed at {origin} h={horizon}: {e}")
                pass

            pbar.update(1)

    pbar.close()

    if len(results) == 0:
        print("   ⚠️  No forecasts generated")
        return None

    df = pd.DataFrame(results)
    output_path = results_dir / 'tabpfn.parquet'
    df.to_parquet(output_path, index=False)
    print(f"✅ Saved {len(df)} forecasts to {output_path}")
    return df

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("FLU FORECASTING - ROLLING WINDOW EVALUATION")
    print("="*60)

    # Run each model
    sarima_df = run_sarima()
    lgb_df = run_lightgbm()
    chronos_df = run_chronos()
    tabpfn_df = run_tabpfn()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"SARIMA+Fourier: {len(sarima_df)} forecasts")
    print(f"LightGBM: {len(lgb_df)} forecasts")
    print(f"Chronos-Tiny: {len(chronos_df)} forecasts")
    if tabpfn_df is not None:
        print(f"TabPFN-TS: {len(tabpfn_df)} forecasts")
    else:
        print(f"TabPFN-TS: Skipped (no API key or SKIP_TABPFN=1)")

    print("\n✅ All forecasts complete!")
    print(f"Results saved to: {results_dir}")

    # Show next steps
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("Run notebook 03_evaluation.ipynb to:")
    print("  • Calculate MAE, MASE, CRPS metrics")
    print("  • Compare models by horizon")
    print("  • Run Diebold-Mariano tests")
    print("  • Generate visualizations")
