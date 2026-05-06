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
