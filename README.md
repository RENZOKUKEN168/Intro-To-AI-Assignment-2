# Assignment 2B: Traffic Prediction and Route Guidance

This repository combines the Assignment 2A tree-based route search with the Assignment 2B traffic prediction pipeline. The A2A search code remains available at the repository root.

## A2B data and models

The supplied SCATS dataset is processed into one record per site every 15 minutes. The project prepares chronological training, validation, and test sequences for lookbacks of 4, 8, and 12 observations, each predicting the next 15-minute traffic count. The model comparison is LSTM, GRU, and BiLSTM; a decision-tree regressor remains available as an optional baseline.

## Setup

Use Python 3.11 for TensorFlow compatibility. Create and activate a virtual environment, then install the ML requirements:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-ml.txt
```

## Reproduce preprocessing and training

```sh
python -m ml.preprocess
python -m ml.dataset
python -m ml.train_all_models
```

Training writes one model per architecture and lookback under `models/`, plus a consolidated `models/evaluation_results.csv` and held-out prediction CSVs. Metrics are reported in original vehicles-per-15-minute units.

## Run tests

```sh
python -m unittest discover -s ml/tests -v
python -m unittest discover -s tbrgs/tests -v
python -m unittest -v test_models
python verify_project.py
```

The model unit suite has 15 cases. The data and route suites add coverage for preprocessing, route-weight conversion, and route search.

## Desktop GUI

Start the graphical application from the repository root after installing dependencies and training the selected model:

```sh
python -m tbrgs.gui
```

The **Traffic forecast** tab predicts one site's next 15-minute flow. The **Route planner** tab runs the A2A A* integration and displays the selected route and estimated directed-link travel times. Route planning requires populated, verified `scats_sites.csv` and `road_links.csv` files; the GUI does not create a road network from the empty templates. The interface uses Python's built-in Tkinter library.

## A2B route integration

The real TBRGS command is `python -m tbrgs.app`; it requires populated `data/tbrgs/scats_sites.csv` and `data/tbrgs/road_links.csv` files. The template files document their schemas. The app predicts flow for the requested departure time, converts it to travel-time edge costs, then calls the A2A A* search. Those real Boroondara road links and their SCATS mapping must be supplied from verified assignment data; the repository's demo network is synthetic and must not be presented as a real route result. See [README_TBRGS.md](README_TBRGS.md).

The A2A command line remains `python search.py <filename> <method>`.
