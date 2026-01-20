# Murmur Detection System

This repository provides a complete, CPU-only baseline system for heart murmur detection with safety, explainability, and deployment-focused outputs. The system is deterministic and runs end-to-end with a single command.

## Quick Start

1. Create the expected data layout:

```
murmur_system/
  data/
    audio/
    labels.csv  (or train.csv + test.csv)
```

2. Install requirements:

```
pip install -r requirements.txt
```

3. Run the full pipeline:

```
python run_all.py
```

4. Launch the demo app:

```
streamlit run src/demo_app.py
```

## Data Format

### Option A: `data/labels.csv`
Required columns:
- `filepath` (relative to `data/audio/` or absolute)
- `label` (0 for normal, 1 for murmur)

Optional columns:
- `patient_id`
- `severity`, `grade`, or `diagnosis`

### Option B: `data/train.csv` and `data/test.csv`
Both files must include `filepath` and `label`.

## Outputs

Artifacts are saved to `outputs/`:

- `outputs/figures/feature_importance.png`
- `outputs/tables/explanations.csv`
- `outputs/tables/quality_report.csv`
- `outputs/figures/quality_vs_error.png`
- `outputs/figures/reliability_diagram.png`
- `outputs/tables/triage_results.csv`
- `outputs/tables/risk_results.csv`
- `outputs/figures/risk_distribution.png`
- `outputs/logs/summary.txt`

## Safety Notes

This system is a baseline and should not be used for clinical diagnosis. Always ensure human review and validation on local datasets.
