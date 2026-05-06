# Architecture Overview

This document captures the initial repository-level architecture. Refine it as the team confirms requirements from the proposal and implementation work.

## System Areas

```text
Nearable device
  ├── Sensors
  ├── Firmware
  └── Communication layer

Software platform
  ├── Mobile app
  ├── Backend API
  ├── Data processing
  └── Dashboard
```

## Hardware

Hardware design files live in `hardware/`.

- `schematics/`: circuit diagrams and wiring references
- `pcb/`: board design files
- `enclosures/`: case and mechanical design
- `datasheets/`: component datasheets and vendor references

## Firmware

Firmware lives in `firmware/`.

- `src/`: production firmware source
- `include/`: shared firmware headers
- `tests/`: firmware unit or hardware-in-loop tests

## Software

Software lives in `software/`.

- `mobile/`: mobile application
- `backend/`: API and services
- `dashboard/`: web dashboard or monitoring interface
- `shared/`: shared schemas, models, or utilities

## Data

Data lives in `data/`.

- `samples/`: small example datasets safe to commit
- `raw/`: raw captured data, ignored by Git by default
- `processed/`: generated or cleaned data, ignored by Git by default

## Interfaces To Define

- Device-to-app communication protocol
- Backend API contract
- Sensor data schema
- Calibration process
- Error and status reporting
- Security and privacy expectations
