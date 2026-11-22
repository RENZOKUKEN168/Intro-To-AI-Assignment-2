# Assignment 2B: Traffic Prediction and Route Guidance

This repository combines the Assignment 2A search algorithms with the Assignment 2B traffic-prediction and route-guidance pipeline.

## Traffic data and models

The original October 2006 SCATS workbook is preserved under `data/raw/`. The processed dataset is stored in `data/processed/scats_traffic_clean.csv`, with one record per SCATS site and 15-minute interval. Shared chronological train/validation/test sequences are prepared for lookbacks of 4, 8, and 12 observations.

The required model comparison is **LSTM, GRU, and Random Forest Regressor**. The existing **BiLSTM** is retained as an additional team model. A Decision Tree regressor remains an optional baseline and is not a substitute for Random Forest.

## Setup

Install the base dependencies (including `xlrd` for the original `.xls` workbook):

```powershell
python -m pip install -r requirements.txt
```

LSTM, GRU and BiLSTM training/inference additionally require TensorFlow in a Python version supported by the TensorFlow release. Use a separate compatible environment if your current Python version is not supported. Random Forest does not require TensorFlow.

## Prepare data and train models

If the processed data and sequence files already exist, do not regenerate them unnecessarily. To reproduce preprocessing and shared sequence creation:

```powershell
python -m ml.preprocess
python -m ml.dataset
```

Train the neural models (TensorFlow environment required):

```powershell
python -m ml.train_all_models
```

Train Random Forest separately so its artifacts do not overwrite neural-model results:

```powershell
python -m ml.train_random_forest --lookbacks 4 8 12 --n-estimators 50 --max-depth 16 --n-jobs -1
```

Random Forest artifacts and evaluation results are written to `models/random_forest/`.

## Tests

Run from the repository root:

```powershell
python -m unittest -v test_models
python -m unittest discover -s ml/tests -v
python -m unittest discover -s tbrgs/tests -v
python verify_project.py
```

The neural inference test requires TensorFlow; the raw `.xls` loading test requires `xlrd`.

## Desktop GUI

Start the graphical application from the repository root after installing dependencies and training the selected model:

```sh
python -m tbrgs.gui
```

The **Traffic forecast** tab predicts one site's next 15-minute flow. The **Route planner** tab runs the A2A A* integration and displays the selected route and estimated directed-link travel times. Route planning requires populated, verified `scats_sites.csv` and `road_links.csv` files; the GUI does not create a road network from the empty templates. The interface uses Python's built-in Tkinter library.

## Assignment 2A route integration

`python -m tbrgs.app` runs Assignment 2A A* using traffic-predicted travel-time edge costs and also calculates up to five distinct loopless alternatives using Yen's algorithm. The SCATS node list `data/tbrgs/scats_sites.csv` is populated with the 40 SCATS IDs and descriptions from the supplied VicRoads site listing. The directed road-link file `data/tbrgs/road_links.csv` still needs verified road connections, measured distances and controlled-intersection counts before a real route can be run.

The traffic-count location catalogue is retained in `data/raw/` for reference, but it does not contain an explicit SCATS-ID field and is not used to invent node coordinates or road links. See [README_TBRGS.md](README_TBRGS.md) and [docs/INTEGRATION_HANDOFF.md](docs/INTEGRATION_HANDOFF.md).

The A2A command line remains `python search.py <filename> <method>`.
