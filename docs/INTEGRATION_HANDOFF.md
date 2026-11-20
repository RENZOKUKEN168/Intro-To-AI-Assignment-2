# Assignment 2B integration handoff and final checks

## Changes made in this package

1. Added `ml/train_random_forest.py`, which trains Random Forest on the existing shared chronological sequence files and writes its own artifacts under `models/random_forest/`.
2. Updated `ml/predict_model.py` to load Random Forest `.joblib` artifacts and expose `RandomForest` as a selectable inference model. Random Forest inference does not import TensorFlow.
3. Updated `tbrgs/app.py` to keep Assignment 2A A* in the integrated workflow and also return up to five loopless routes using Yen's algorithm. It checks that A* and Yen agree on the best travel-time cost.
4. Added tests for Random Forest prediction loading and for the combined A*/Yen application workflow.
5. Added the supplied flow-to-travel-time PDF and traffic-count location catalogue to the project sources.
6. Updated the model and routing READMEs with commands and unresolved assumptions.

## What is still not supplied by the source files

- `data/tbrgs/scats_sites.csv` now contains the 40 SCATS IDs in the processed traffic data and their descriptions from the supplied VicRoads site listing. Coordinates are blank because that listing supplies map references rather than latitude/longitude. `data/tbrgs/road_links.csv` is still missing; only its template is included. Do not fabricate links or distances.
- The traffic-count location catalogue does not have an explicit SCATS-ID field, so it cannot safely be used to generate the SCATS road graph.
- No GUI source was present in the uploaded repository snapshot. If a teammate has a GUI branch, merge/integrate it separately.
- The conversion guide uses origin-site flow, while another assignment note says destination-site flow. Confirm which rule the lecturer expects.
- The conversion guide's 1,500 vehicles/hour capacity is illustrative. The app stops on over-capacity predictions by default; obtain approval before using the `clamp` option.
- LSTM/GRU/BiLSTM inference requires TensorFlow in a supported Python environment. Random Forest does not.

## Run these checks in order

From the repository root in PowerShell:

```powershell
python -m py_compile ml/predict_model.py ml/train_random_forest.py tbrgs/app.py
python -m unittest -v test_models
python -m unittest discover -s ml/tests -v
python -m unittest discover -s tbrgs/tests -v
python verify_project.py
```

Then train Random Forest:

```powershell
python -m ml.train_random_forest --lookbacks 4 --n-estimators 5 --max-depth 16 --n-jobs 1
python -m ml.train_random_forest --lookbacks 4 8 12 --n-estimators 50 --max-depth 16 --n-jobs -1
```

Finally, after verified road-network CSVs are supplied, run a real route test:

```powershell
python -m tbrgs.app 2000 3002 "2006-10-29 04:00" --model RandomForest --lookback 12 --k-routes 5 --skip-unavailable-sites
```

This final route command cannot pass until the verified `road_links.csv` is added and the origin and destination have enough history. With `--skip-unavailable-sites`, other sites lacking consecutive history are excluded together with their incident links; the resulting network may be less connected. If the predicted hourly flow exceeds the illustrative capacity, the default policy will stop and report it for review.

## Validation performed for this packaged copy (17 November 2026)

The following checks were run against the files in this folder:

- `python -m py_compile models.py ml/predict_model.py ml/train_random_forest.py tbrgs/app.py tbrgs/travel_time.py tbrgs/top_k_routes.py` — passed.
- `python -m ml.train_random_forest --help` — passed.
- `python -m unittest -v test_models` — 23 tests passed.
- `python -m unittest discover -s tbrgs/tests -v` — 31 tests passed, including the up-to-five-route integration tests.
- `python verify_project.py` — 114 Assignment 2A checks passed.
- Random Forest training — completed for lookbacks 4, 8 and 12, using 50 trees and max depth 16. Artifacts are included in `models/random_forest/`.
- Random Forest inference — successful for SCATS 2000 and 3002 at `2006-10-29 04:00` for all three lookbacks.

Random Forest held-out test metrics (original vehicles per 15-minute units):

| Lookback | MAE | RMSE | R² | NRMSE |
|---:|---:|---:|---:|---:|
| 4 observations | 28.57 | 42.73 | 0.9753 | 0.0308 |
| 8 observations | 27.18 | 40.32 | 0.9780 | 0.0290 |
| 12 observations | 26.63 | 39.61 | 0.9786 | 0.0285 |

The additional `ml/tests` suite was attempted in this packaging environment. One original-workbook test could not run because `xlrd` is not installed in this environment, and one TensorFlow-dependent LSTM inference test was skipped because TensorFlow is not installed. Both dependencies are documented in the requirements files. Install the base requirements and rerun this suite on the team's machine.

## Final blockers to check with the team

1. **Real road network:** `data/tbrgs/road_links.csv` is not present. Only `road_links_template.csv` is included. No real route can be claimed until directed SCATS-to-SCATS connections, measured road distances and controlled-intersection counts are verified and entered.
2. **GUI:** no GUI source was present in the uploaded repository snapshot. Add the teammate's GUI branch separately if one exists; do not present the command-line tool as the required GUI.
3. **Flow-source assumption:** the supplied conversion guide says to use the starting SCATS site's flow, while another assignment note says destination-site flow. Confirm the expected rule and set `--flow-source` explicitly.
4. **Capacity assumption:** the guide's 1,500 vehicles/hour capacity is illustrative. The default `raise` policy intentionally stops when a prediction exceeds it. Do not change to `clamp` unless approved and documented.

The included Random Forest artifacts are trained outputs, not a substitute for re-running model evaluation in the team's target environment. The route smoke test remains blocked by item 1 above.
