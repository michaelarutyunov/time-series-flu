# Forecast Configuration

## Optimized Setup (Current)

The rolling forecast evaluation has been optimized for API efficiency while maintaining statistical validity.

### Configuration

```python
HORIZONS = [7, 28]  # 7-day and 28-day ahead forecasts
ORIGINS = pd.date_range('2024-07-08', '2025-05-26', freq='2W-MON')  # Bi-weekly
MIN_TRAIN = 730  # 2 years minimum training
```

### Key Metrics

- **Total forecast origins**: 24 (bi-weekly Mondays)
- **Horizons per origin**: 2 (7-day and 28-day ahead)
- **Total forecasts per model**: 48
- **Time period covered**: 11 months (July 2024 - May 2025)
- **API calls (TabPFN)**: 48 (fits free tier comfortably)

### Why This Configuration?

**Horizons: [7, 28]**
- **7-day**: Most practical for operational decisions (weekly reporting cycles)
- **28-day**: Strategic planning horizon (monthly forecasts)
- Dropped 1-day (too easy) and 14-day (middle ground, less critical)

**Bi-weekly origins (24 points)**
- Statistically robust sample size (>20 recommended minimum)
- Captures seasonal variation across full flu cycle
- Reduces API calls by 50% vs weekly
- Still provides weekly cadence via 7-day forecasts

**Time period (11 months)**
- July 2024 - May 2025
- Includes full flu season peak (Winter 2024/25)
- Requires 730 days training data (starts 2024-07-08)

### API Call Breakdown

| Model | Calls per Origin | Total Calls | Runtime |
|-------|-----------------|-------------|---------|
| SARIMA+Fourier | 2 | 48 | ~2 min |
| LightGBM | 2 | 48 | ~1 min |
| Chronos-Tiny | 2 | 48 | ~5 min |
| TabPFN-TS | 2 | 48 | ~10-20 min* |

*Depends on API response time

### Running the Forecasts

**Option 1: Standalone script (recommended)**
```bash
source .venv/bin/activate
python run_forecasts.py
```

To skip TabPFN (if no API key):
```bash
SKIP_TABPFN=1 python run_forecasts.py
```

**Option 2: Jupyter notebook**
```bash
jupyter notebook nb/02_roll_loop.ipynb
```

### Output Files

All forecasts saved to `results/forecasts/`:
- `sarima_fourier.parquet` (48 forecasts)
- `lightgbm.parquet` (48 forecasts)
- `chronos.parquet` (48 forecasts)
- `tabpfn.parquet` (48 forecasts, if run)

Each file contains:
- `origin`: When forecast was made
- `date`: Target date being predicted
- `horizon`: Days ahead (7 or 28)
- `model`: Model name
- `q0.1, q0.5, q0.9`: Prediction quantiles (80% interval)
- `actual`: True observed value

## Alternative Configurations

If you need to adjust the configuration:

### More forecasts (better statistics)
```python
ORIGINS = pd.date_range('2024-07-08', '2025-05-26', freq='W-MON')  # Weekly
HORIZONS = [7, 14, 28]  # 3 horizons
# Total: 47 origins × 3 horizons = 141 calls
```

### Fewer forecasts (faster testing)
```python
ORIGINS = pd.date_range('2025-03-01', '2025-06-01', freq='W-MON')  # 3 months
HORIZONS = [7, 28]
# Total: ~13 origins × 2 horizons = 26 calls
```

### Focus on flu season
```python
ORIGINS = pd.date_range('2024-12-01', '2025-02-28', freq='W-MON')  # Winter only
HORIZONS = [7, 14, 28]
# Total: ~13 origins × 3 horizons = 39 calls
```

## Statistical Validity

With 24 bi-weekly origins:
- ✅ Adequate sample size (>20 recommended)
- ✅ Covers full seasonal cycle
- ✅ Can detect meaningful differences between models
- ✅ Suitable for LinkedIn article or blog post
- ⚠️ For academic paper, consider weekly origins (47 points)

## Next Steps

After generating forecasts, proceed to:
1. **03_evaluation.ipynb**: Calculate metrics (MAE, MASE, CRPS)
2. Compare models by horizon
3. Run Diebold-Mariano statistical tests
4. Generate visualizations for article
