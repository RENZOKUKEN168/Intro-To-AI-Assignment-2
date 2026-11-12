# Member 2: GRU and BiLSTM for Traffic Forecasting

This implementation follows the project forecasting setup: one traffic observation every 15 minutes; input windows of 4, 8, or 12 observations (1, 2, or 3 hours); predict the next 15-minute observation (`window + 1`). The model comparison should include LSTM (Member 1), GRU, and BiLSTM. A regression-tree implementation is also kept as an optional baseline.

## Data handoff

Prepare one time-ordered series per traffic location. Sort each location by timestamp and call `make_forecasting_windows` separately for each location so a window never crosses between sites. The function accepts a 1-D traffic-flow series or a 2-D array shaped `(observations, features)` and returns:

- `X`: `(samples, lookback, features)`
- `y`: the selected traffic-flow value at the next observation

Use the same chronological train/validation/test partitions for all models. Fit any scaling on training data only.

## Train GRU / BiLSTM

Use Python 3.11 on macOS, select the project virtual environment in VS Code, then install dependencies:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-gru.txt
```

Example:

```python
from models import GRURegressor, BiLSTMRegressor, evaluate_predictions, make_forecasting_windows

X, y = make_forecasting_windows(site_flow, lookback=4)  # 1 hour of history, predict next 15 minutes
model = GRURegressor()  # Or BiLSTMRegressor()
model.fit(X_train, y_train, validation_data=(X_validation, y_validation), epochs=50)
predictions = model.predict(X_test)
print(evaluate_predictions(y_test, predictions).as_dict())
```

Train/evaluate with lookbacks `4`, `8`, and `12` separately so the results compare one, two, and three hours of history.

## Evaluation metrics

The evaluator reports RMSE, MAE, MSE, and R² as shown in the project slide. It also reports NRMSE for the assignment comparison; here NRMSE is RMSE divided by the range of held-out actual values. Compare every model on the same held-out timestamps and target values.

## Tests

From the repository root, run:

```sh
python -m unittest -v test_models
```

The tests cover all three lookback sizes, next-step target alignment, metrics, input validation, regression-tree save/load, and GRU/BiLSTM dependency behavior.
