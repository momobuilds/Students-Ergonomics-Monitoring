# Nearable Project

Team repository for the Nearable project. This repo is organized so hardware, firmware, software, documentation, and data work can move in parallel without stepping on each other.

## Repository Map

```text
.
├── assets/              # Images, diagrams, and project media
├── data/                # Sample, raw, and processed datasets
├── docs/                # Proposal, architecture, research, reports, meetings
├── firmware/            # Embedded code and firmware tests
├── hardware/            # Schematics, PCB files, enclosures, datasheets
├── scripts/             # Reusable project automation
├── software/            # Mobile app, backend, dashboard, shared code
└── tests/               # Cross-system and integration tests
```

## Quick Start

1. Read the project proposal at [docs/proposal/project-proposal.pdf](docs/proposal/project-proposal.pdf).
2. Review [docs/architecture/overview.md](docs/architecture/overview.md) to understand the planned system.
3. Pick work from [docs/project-board.md](docs/project-board.md) or create an issue.
4. Work on a branch and open a pull request for review.

## Team Workflow

- Use short-lived feature branches: `feature/<topic>`, `fix/<topic>`, `docs/<topic>`.
- Keep pull requests focused on one area when possible.
- Update documentation when a decision changes the system design, interfaces, hardware choices, or setup process.
- Put meeting notes in `docs/meetings/YYYY-MM-DD-topic.md`.
- Keep raw experimental data in `data/raw/` and derived outputs in `data/processed/`.

## Main Work Areas

- `hardware/`: sensor selection, schematics, PCB, case/enclosure design.
- `firmware/`: sensor reading, device communication, power management, embedded tests.
- `software/backend/`: APIs, storage, authentication, processing services.
- `software/mobile/`: user-facing mobile experience.
- `software/dashboard/`: web dashboard or monitoring interface.
- `docs/`: proposal, research notes, requirements, architecture, reports.

## Status

Initial repository structure created on 2026-05-06.

## Simulated Sensor Fusion Prototype

This prototype provides a complete simulated-data workflow for the ergonomics monitoring system. It generates two hours of minute-level fake sensor readings, applies feature-level and decision-level sensor fusion, explains the result with a local LLM advisor interface, and displays everything in a Streamlit dashboard.

The project can run without real sensors. The current fake dataset includes:

- `heart_rate_bpm`
- `baseline_heart_rate_bpm`
- `blink_rate_per_min`
- `posture_status`
- `slouching_time_minutes`
- `temperature_c`
- `data_source`

Install the project requirements:

```bash
pip install -r requirements.txt
```

Generate the fake sensor dataset:

```bash
python Scripts/SensorFusion/generate_fake_data.py
```

Run the dashboard:

```bash
streamlit run Scripts/SensorFusion/dashboard_app.py
```

The sensor fusion formula is:

```text
final_risk_score =
0.35 * posture_score
+ 0.30 * blink_score
+ 0.25 * heart_score
+ 0.10 * temperature_score
```

The final status is `Normal` when the score is below `0.30`, `Warning` when the score is from `0.30` up to but not including `0.60`, and `High Risk` when the score is `0.60` or above.

Later, the simulated CSV will be replaced by live values from the heart-rate wearable, webcam blink detector, posture detector, and optional temperature patch. Teammates should keep the same column names when connecting real sensor modules so `Scripts/SensorFusion/fusion.py` and the dashboard can continue using the same interface.

The local LLM advisor does not calculate or override the risk score. The deterministic fusion algorithm calculates the score and status first, and the advisor only explains that result and gives student-friendly ergonomic recommendations.
