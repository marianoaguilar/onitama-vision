# TFG-Onitama

TFG-Onitama is an academic final project that brings together a complete **Onitama** rules engine, a search-based AI, and a computer-vision-assisted desktop application for playing on a **real physical board**.

The project is designed for a specific use case: a human plays Onitama on a physical board, a camera observes the board and cards, the software reconstructs the game state, validates legal transitions, and the AI answers with its own move. The main user-facing entry point is the desktop GUI, but the repository also includes supporting tools for testing, calibration, benchmarking, and experimentation.

## Table of Contents

- [Implemented features](#implemented-features)
- [Main technologies](#main-technologies)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running the project](#running-the-project)
- [Project structure](#project-structure)
- [Limitations](#limitations)
- [Documentation](#documentation)
- [Legal notice](#legal-notice)

## Implemented features

- Complete Onitama rules engine with immutable game state, official card handling, and legal move generation.
- Search-based AI opponent with negamax, alpha-beta pruning, quiescence search, transposition-table support, and multiple heuristic evaluators.
- Vision pipeline that detects board pieces, classifies the five visible cards, and reconstructs a snapshot from camera frames.
- Integration layer that stabilizes repeated observations, validates legal one-ply transitions, and coordinates turns.
- Desktop GUI with integrated board calibration, card-ROI calibration, live board/card rendering, status feedback, and optional camera preview.
- Auxiliary tooling for automated tests, vision debugging, data capture, tournaments, and search benchmarking.

## Main technologies

- Python
- `PySide6` for the desktop GUI
- `numpy` and `opencv-python` for image processing
- `ultralytics` for YOLO-based detection and classification

## Requirements

- Python `3.10` or newer
- A webcam or compatible camera
- A physical Onitama board and cards

## Installation

Clone the repository:

```bash
git clone https://github.com/marianoaguilar/TFG-Onitama.git
cd TFG-Onitama
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the main application dependencies:

```bash
pip install -e .[gui,vision]
```

If you also want the test dependency and the full optional stack:

```bash
pip install -e .[full]
```

The repository already includes the trained models in `models/`. Calibration files live under `data/vision/`, but a different camera or physical setup will usually require recalibration.

## Running the project

### Desktop application

This is the main way to use the project:

```bash
onitama-vision
```

You can also run it directly from the source tree:

```bash
PYTHONPATH=src python -m onitama.gui.vision_app
```

### First-time setup

From the GUI you can:

- choose whether the human plays as red or blue,
- choose the AI difficulty,
- calibrate the board area,
- calibrate the card regions,
- start a new game,
- open a live camera window.

If the camera, table position, board placement, or card layout changes, recalibration is usually required.

Recommended physical setup:

![Ideal physical setup](docs/Disposicion_ideal.png)

### Development and evaluation tools

Run the test suite:

```bash
pytest -q
```

Tournament script:

```bash
python scripts/tournament.py --help
```

Search benchmark:

```bash
python scripts/bench_search.py --help
```

## Project structure

```text
TFG-Onitama/
├── src/onitama/
│   ├── engine/        # Game rules, state, cards, pieces, actions
│   ├── ai/            # Search, evaluation, AI controllers
│   ├── vision/        # Board detection, card classification, snapshots
│   ├── integration/   # Stabilization and legality synchronization
│   ├── gui/           # Desktop application
│   ├── cli/           # Auxiliary terminal interfaces
│   └── app/           # Shared runtime models and orchestration
├── scripts/           # Calibration, debugging, tournaments, benchmarks
├── tests/             # Automated test suite
├── models/            # Trained vision models
├── data/vision/       # Calibration and ROI data
└── docs/              # Project notes and supporting documentation
```

## Limitations

- The vision pipeline depends on camera quality, lighting, framing, and physical setup consistency.
- Calibration is mandatory for reliable vision-assisted play.
- The system is designed around the included models and expected board/card layout.
- The GUI is the primary supported play mode; some CLI tools exist mainly for development and debugging.

## Documentation

For a full technical explanation of the project, methodology, architecture, and implementation details, see [docs/memoria/proyecto.pdf](/home/mariano/Escritorio/TFG-Onitama/docs/memoria/proyecto.pdf).

## Legal notice


This repository is an academic project and is not an official Onitama product.

Onitama, its name, rules, artwork, branding, and related assets belong to their respective rights holders. This project does not claim ownership over them.
