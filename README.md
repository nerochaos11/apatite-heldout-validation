# Apatite held-out validation — code and data

Supporting material for the manuscript

> **Sample and source constraints on ore-deposit discrimination using apatite trace elements**
> Jianshe Zhang, Caiyu Xiang, Hongze Chen, Zhijian Ni, Haiying Wang
> Submitted to *Computers & Geosciences*

---

## Requirements before you run anything

| Item | Value |
|---|---|
| Operating system | Windows 10/11, or Linux/macOS with equivalent packages |
| Python | **3.14.5** (recorded in `05_environment/environment_versions.json`) |
| RAM | 8 GB minimum; 16 GB recommended for the retrain path |
| GPU | not required |
| Disk | about 2.5 GB free for a full retrain with recomputed outputs |
| Network | not needed for any analysis step. Needed only (i) to load the satellite basemap of the interactive map, and (ii) by the browser-screenshot tool (`04_analysis_code/map_capture/`) |

Install the pinned dependencies:

```text
python -m pip install -r 05_environment/requirements-science.txt
```

Figure redrawing additionally needs `requirements-figures.txt`; the interactive
map screenshot tool needs `requirements-map.txt` and a Node.js runtime with
Microsoft Edge.

**Quick check without installing anything beyond the standard library:**

```text
python verify.py
```

This reads the retained tables, recomputes every reported metric from the saved
out-of-fold probabilities, and checks the SHA-256 of every file listed in
`file_checksums.csv`. It does not train, overwrite results, or load the model.

---

## What the study asks

Public apatite trace-element compilations are organised as analyses nested in
registered samples, samples nested in deposits, and deposits nested in
literature sources. All previous apatite classifiers evaluate their models by
splitting records at random, which places near-duplicate analyses from the same
sample on both sides of the split. This package implements a **four-level
holdout protocol** that separates four different evaluation targets inside a
single pipeline:

| Level | Unit held out |
|---|---|
| record | stratified random split |
| registered sample | all analyses of one sample group |
| deposit name | all sample groups sharing a normalised name |
| literature source | all records from one publication or data release |

## Main results

Six classes, pooled out-of-fold macro-F1 (LightGBM, seed 42, n = 6,502):

| Holdout level | macro-F1 | macro ROC-AUC |
|---|---|---|
| record | 0.8411 | 0.9749 |
| registered sample | 0.7377 | 0.9467 |
| literature source | 0.3079 | 0.6577 |

Mineralized five classes (n = 3,152), ten-seed mean macro-F1:
0.8884 (record), 0.8194 (sample), 0.4414 (deposit name), 0.3632 (source).

Every source-held-out training fold retains all six target classes, so complete
class coverage does not ensure transfer to an unseen source.

---

## Directory layout

```text
apatite_heldout_validation/
  README.md                        this file
  verify.py                        read-only metric recomputation and checksum check
  file_checksums.csv               SHA-256 of every supplied file
  version_info.json                data, model and map versions

  01_data_and_results/             frozen tables, screening decisions, folds, OOF, results
      main_data_clean.csv          6,502 modelling records (the primary table)
      candidates_merged.csv        6,708 candidate records before screening
      per_record_screening.csv     one row per candidate, with the exclusion decision
      OOF/                         106 out-of-fold experiments (probabilities, folds, metrics)
      LOSO/                        90 leave-one-source-out results
      *                            sensitivity branches, calibration, SHAP, variance

  02_source_data/                  merged input, upstream compiled tables, USGS release,
                                   application inputs, historical comparison
  03_figure_data/                  panel plotting data (Fig01-Fig17, see Figure data)
                                   and Table01-Table08 CSVs
  04_analysis_code/                analysis modules (see below)
  05_environment/                  pinned dependency files and recorded versions
  06_model_and_map/                full-data refit model, interactive Leaflet map
                                   (Leaflet is bundled inline; only the Esri
                                   satellite basemap needs a network connection)
```

### Analysis modules

| File | Role |
|---|---|
| `prepare_data.py` | screening, deduplication, grouped summaries |
| `experiment_core.py` | grouped fitting and pooled out-of-fold evaluation |
| `figures.py` | statistical figures |
| `supplementary.py` | SHAP, calibration, nested variance |
| `export_figure_data.py` | writes `03_figure_data/` |
| `application.py` | application datasets and interactive map |
| `map_figure.py` | static map rendering |
| `verify_variance.py` | REML cross-check |
| `configuration_sensitivity.py` | alternative preprocessing configuration |
| `case_and_sensitivity.py` | Pea Ridge case and Zirner2015 branch |
| `collate.py` | metric aggregation |
| `paths.py` | path definitions |
| `run_all.py` | driver |
| `map_capture/` | browser screenshot tool (Node.js + Edge) |

---


## Figure data

`03_figure_data/` follows the **17-panel layout** used by the plotting code
(`figures.py` and `map_figure.py`). The manuscript reports **15 figures**,
because two published figures combine two panels each. The mapping is:

| Published figure | Panel folders in `03_figure_data/` |
|---|---|
| Fig. 1 | Fig01 |
| Fig. 2 | Fig02 |
| Fig. 3 | Fig03 |
| Fig. 4 | Fig04 |
| **Fig. 5** | **Fig05 + Fig08** |
| Fig. 6 | Fig06 |
| Fig. 7 | Fig07 |
| Fig. 8 | Fig09 |
| Fig. 9 | Fig10 |
| Fig. 10 | Fig11 |
| Fig. 11 | Fig12 |
| Fig. 12 | Fig13 |
| Fig. 13 | Fig14 |
| **Fig. 14** | **Fig15 + Fig16** |
| Fig. 15 | Fig17 |

The panes of the two combined figures were assembled and relabelled at layout
stage, outside the plotting code; the underlying plotting data are complete for
every pane in the folders above. `figure_data_index.csv` lists every file with
its folder and its panel subject.

---

## How to rerun

```text
python verify.py --retrain              full pipeline; results go to recomputed/
python verify.py --redraw-statistics    redraw statistical figures from saved results
python verify.py --redraw-map           redraw the map figure from the bundled view
```

Nothing is overwritten: all output goes to `recomputed/` and the frozen tables
stay as supplied. The main seed is 42; the repeated-validation seeds are 42, 7,
13, 21, 33, 99, 123, 2024, 555 and 888.

---

## Key conventions

- `input_row` links a record to its screening decision in `per_record_screening.csv`
  and to its out-of-fold probability in `OOF/`.
- `sample` is the registered sample key, built as *source + registered sample name*.
- `source` is the compilation source identifier; `deposit_group` is the
  normalised deposit-name text group.
- In OOF files, `fold` is the outer test fold and `p_<class>` is the out-of-fold
  probability for that class.
- `*_folds.csv` files give fold composition; `*_metrics.json` files give the
  corresponding metrics.


## Archived artefacts not produced by the plotting scripts

Three files in `01_data_and_results/` and `02_source_data/historical/` are
convenience snapshots rather than pipeline outputs. They are supplied because
the manuscript refers to the quantities they hold.

| File | Content |
|---|---|
| `01_data_and_results/model_input_matrix.npz` | frozen model input for the 6,502-record main cohort. Arrays: `X` (6,502 x 9, the nine predictors in the order given in `model_version.json`), `elements` (6,502 x 6 raw concentrations), `y` (class label), `sample`, `source`, `deposit_group`, `input_row`, and `features` (the predictor names). Loading this file reproduces the inputs of the fitted model without rerunning preprocessing. |
| `02_source_data/historical/example_site_11_3522_predictions.csv` | predictions of the historical 3,522-record model for the eleven example-site analyses of Fig. 14c-d, with coordinates, source fields and the uncalibrated class probabilities. Model version `LGBM-main3522-refit-20261020`. |
| `01_data_and_results/config_sensitivity_excel_*_s42.csv` | per-record out-of-fold probabilities of the alternative preprocessing configuration described in Table 8, for the record, sample and source holdout schemes at seed 42. Generated by `04_analysis_code/configuration_sensitivity.py`. |

## Data provenance

Original datasets belong to their publishers and are cited as follows:

| Dataset | Source | Identifier |
|---|---|---|
| Apatite Classifier v2 | Qiu et al. (2022, 2024) | doi:10.5281/zenodo.7094895 |
| Expanded apatite compilation | Zheng et al. (2024) | doi:10.2138/am-2023-9115 |
| Pea Ridge / regional igneous | Mercer and Koenig (2019); Mercer et al. (2020) | doi:10.5066/P9YIHMO8 |
| Fosterville application data | Tylkowski et al. (2025) | doi:10.17632/jt26hpy446.1 |

The 90 source identifiers follow the compilation's own fields and do not equal
90 independently verified deposits, laboratories or mineralising events. See
`02_source_data/sources_and_scope.md` for the full source inventory and
`01_data_and_results/appendix_90_sources.csv` for the record counts per source.

## Note on language

All directory names, file names, column headers and documentation in this
package are in English. One file, `01_data_and_results/main_data_clean.csv`, is
supplied byte-for-byte as originally produced so that its recorded SHA-256 stays
valid; three of its columns keep the original Chinese status and provenance
values. They are explained in `01_data_and_results/field_descriptions.md`.


## Related repositories

| Repository | URL |
|---|---|
| Source code (live, MIT licence) | https://github.com/nerochaos11/apatite-heldout-validation |
| This archive | https://doi.org/10.5281/zenodo.24243154 |

The GitHub repository and this archive contain the same analysis modules. The
archive additionally holds the frozen tables, the out-of-fold probabilities and
the fitted model, so it can be verified offline without network access.

## Licence

Code: MIT. Data: see the upstream licences of the four publishers above.
