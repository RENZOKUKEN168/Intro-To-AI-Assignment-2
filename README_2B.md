# Traffic Forecasting Models (Assignment 2B)

The forecasting pipeline uses one observation every 15 minutes and predicts the next 15-minute traffic-flow value. It supports lookback windows of 4, 8, and 12 observations (1, 2, and 3 hours).

## Models

The required comparison set is:

- **LSTM**
- **GRU**
- **Random Forest Regressor**

The existing **BiLSTM** implementation is retained as an additional team model. The Decision Tree regressor is retained as an optional baseline and is not counted as Random Forest.

All models use the same chronological train/validation/test splits and the shared scaler fitted on training data only. Sequence windows are generated independently per SCATS site and reject missing 15-minute intervals, so a window does not cross sites or time gaps.

## Data and preprocessing

The source workbook is `data/raw/Scats Data October 2006.xls`. The processed data expected by the pipeline is `data/processed/scats_traffic_clean.csv`, with columns:

```text
scats_id,timestamp,traffic_flow
```

The prepared sequence files and scaler parameters are under `data/processed/`. Do not edit the original workbook or overwrite the shared model evaluation results when training Random Forest.

To regenerate the shared sequence files only when necessary:

```powershell
python -m ml.dataset
```

## Train Random Forest

Random Forest has a separate training script and writes only to `models/random_forest/`:

```powershell
python -m ml.train_random_forest --lookbacks 4 8 12 --n-estimators 50 --max-depth 16 --n-jobs -1
```

For a quick smoke test before full training:

```powershell
python -m ml.train_random_forest --lookbacks 4 --n-estimators 5 --max-depth 16 --n-jobs 1
```

Outputs include `window_4.joblib`, `window_8.joblib`, `window_12.joblib`, a separate `evaluation_results.csv`, and test prediction CSVs. These do not overwrite LSTM, GRU, or BiLSTM artifacts.

## Train neural models

LSTM, GRU, and BiLSTM use TensorFlow/Keras. Install a TensorFlow-compatible Python version and dependencies from `requirements-gru.txt`, then run:

```powershell
python -m ml.train_all_models
```

Random Forest does not require TensorFlow. Its training and inference use scikit-learn and joblib.

## Evaluation

Compare the same held-out timestamps and target values using MAE, MSE, RMSE, R², and NRMSE. NRMSE is RMSE divided by the range of held-out actual values. Predictions are converted back to vehicles per 15-minute interval before metrics are calculated.

## Tests

```powershell
python -m unittest -v test_models
python -m unittest discover -s ml/tests -v
```
