# WAF-Uber: Dynamic Driver–Request Matching

A small, reproducible study of whether historical demand can improve ride dispatch decisions. We compare pickup-time batch matching with a scenario-based lookahead heuristic using recorded NYC Uber requests and a simulated fixed fleet.

**These are real-trip-driven simulation results, not measurements of Uber’s actual service rate or a reconstruction of its dispatch system.**

## Research question and methods

Can dispatch improve future service opportunities by considering both where assigned drivers finish their trips and where unassigned drivers remain?

- **Baseline:** maximize the number of feasible current matches, then minimize total pickup time.
- **Lookahead:** start from the baseline and evaluate driver substitutions and two-order swaps. Minimize current pickup time minus a weighted estimate of future feasible matches, using two historical demand scenarios over the next 30 minutes.
- **Common rules:** match every minute; each driver serves one trip at a time; total request-to-pickup waiting must meet the waiting limit. Unassigned drivers wait in place. There is no pooling, proactive relocation, pricing, or driver entry/exit.

The search keeps the baseline’s selected current request subset and considers at most 20 alternatives over two passes. Future evaluation includes idle drivers and drivers released after trip completion, counting at most one subsequent request per driver. It is a limited heuristic, not a globally optimal dispatch policy or a reproduction of a published reinforcement-learning model.

## Data and evaluation design

Source: [NYC TLC trip records](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page), January 2024 High Volume FHV data. See the [official field dictionary](https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_hvfhs.pdf).

| Item | Setting |
| --- | --- |
| Records | Uber (`HV0003`), both shared flags equal to `N` |
| Area | Both endpoints in Manhattan zones 161, 162, 163, 164, 170, 233 |
| Request window | 17:00–19:00, New York local time |
| Sampling | Reproducible approximately 10% sample |
| Historical period | Nine weekdays, January 2–12 |
| Validation | January 16–17 |
| Test | January 22–24 |
| Simulated fleets | 8 and 12 drivers; two initializations per day |
| Weight selection | 0, 3, or 10; selected on validation and fixed for testing |
| Primary waiting limit | 8 minutes |

Historical OD median passenger-trip times approximate both trip and pickup travel. These estimates are fixed before testing; observed test completion times do not determine simulated driver availability. Policies use identical requests and initial locations. Source metadata, checksums, filtering counts, and travel-table provenance are saved with the data.

## Recorded results

The primary test pools three test days and two initializations per fleet. Waiting time is measured among served requests.

| Fleet | Baseline service rate | Lookahead service rate | Gain (percentage points) | Mean wait change |
| --- | ---: | ---: | ---: | ---: |
| 8 | 30.75% | 34.49% | +3.74 | +1.88 seconds |
| 12 | 41.71% | 46.52% | +4.81 | +2.24 seconds |

Both fleets selected weight 10. The source table is [comparison.csv](results_real/20260909_194042_679086_test/comparison.csv).

An **exploratory sensitivity analysis added after inspecting the primary results** keeps requests, travel estimates, and initializations fixed while repeating validation and testing at waiting limits of 8, 10, and 12 minutes. At 10 minutes, both fleets select weight zero and reproduce the baseline. At 12 minutes, the gains are zero for 8 drivers and 0.53 percentage points for 12 drivers. See [all sensitivity results](results_real/sensitivity_20260909_194334_782980/sensitivity_effects.csv).

The benefit is therefore conditional on the setting. Three test days are insufficient for strong general claims; the sensitivity analysis reuses those days and is not independent confirmation.

## Reproduce the experiments

Use Python 3.10 or newer. Run these commands from the repository directory. No API key is required.

```sh
python -m pip install -r requirements-real.txt
python simulator.py --self-test
python -m unittest discover -s tests -v
python pipeline.py download
python pipeline.py prepare
python pipeline.py smoke
python pipeline.py validate
python pipeline.py test
python pipeline.py report
```

`smoke` is a small execution check, not the final evaluation. Alternatively, `python pipeline.py all` runs download, preparation, validation, testing, and reporting. New experiment runs receive separate directories. Configuration and implementation fingerprints protect against reusing incompatible validation selections.

For the waiting-limit sensitivity analysis, after preparing the primary data:

```sh
python sensitivity.py
```

This reuses the same prepared inputs. Do not change the waiting limit and re-prepare data to reproduce this analysis, since preparation also uses the waiting limit to define its historical travel-data window.

The original monthly Parquet is stored with **Git LFS**. After cloning, use `git lfs install` and `git lfs pull` to retrieve it, or use the download step to obtain the official source. Allow several GB of free disk space. Repository access is required while this project is private.

## Repository guide

| Path | Purpose |
| --- | --- |
| `pipeline.py` | Real-data experiment entry point |
| `experiment_config.json` | Data split and experiment parameters |
| `real_experiment/` | Downloading, preparation, policy evaluation, and reporting |
| `simulator.py` | Core simulator and original synthetic demonstration |
| `sensitivity.py` | Waiting-limit sensitivity experiment |
| `tests/` | Data isolation, solver, and pipeline checks |
| `data/raw/` | Official source files and provenance |
| `data/processed/` | Retained requests, historical estimates, and audit reports |
| `results_real/` | Validation selections, test metrics, and detailed logs |
| `README_SYNTHETIC_CN.md` | Archived Chinese synthetic-demo instructions |
| `VERIFICATION_REAL.md` | Historical verification record from initial delivery |

The repository retains early synthetic outputs, smoke checks, and multiple run directories for traceability. They should not be pooled with the primary test results linked above. The project evolved from synthetic checks to recorded demand and historical travel estimates, followed by validation-selected testing and exploratory sensitivity analysis.

## Interpretation and limitations

TLC records cover recorded trips, not all unsuccessful demand or complete driver online trajectories. Fleet sizes and initial locations are simulation assumptions. Geographic filtering and sampling change supply–demand competition. Zone-level median travel times omit within-zone variation and congestion dynamics. Lookahead omits current unassigned backlog from future scenarios and considers only one subsequent service per driver. These simplifications support a controlled comparison, but limit transfer to operational Uber dispatch.
