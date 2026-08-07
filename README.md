# <img src="assets/logo.svg" alt="logo" width="128" height="128" align="middle"> Kneeliverse

![PyPI - Version](https://img.shields.io/pypi/v/kneeliverse)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/kneeliverse)
![GitHub License](https://img.shields.io/github/license/mariolpantunes/knee)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/mariolpantunes/knee/main.yml)
![GitHub last commit](https://img.shields.io/github/last-commit/mariolpantunes/knee)

**Kneeliverse** is a universal knee/elbow detection library for performance curves.

Estimating the knee of a performance curve is a hard problem, yet the point it
identifies is usually the one that matters: the compromise where further cost
stops buying meaningful performance. This library brings the well-known
detectors, their multi-knee generalisations, and the pre- and post-processing
they need under one consistent API.

## Features

* **Single-knee detectors**: Discrete Curvature, DFDT, Kneedle, L-method,
  Menger curvature and AutoElbow, each exposed as `knee(points) -> int`.
* **Parameter-free detection** *(new in 1.2.0)*: `autoelbow` scores each point
  by a ratio of squared distances to three fixed references and takes the
  largest. No threshold, no sensitivity, no smoothing window - the answer is a
  property of the curve alone, and it handles all four orientations.
* **Multi-knee detection**: Kneedle, Fusion and the Z-method detect multiple
  knees natively; `multi_knee` generalises *any* single-knee function into a
  recursive multi-knee one, so "multi L-method" costs nothing extra.
* **Curve simplification**: a custom RDP that reduces a discrete point set while
  keeping reconstruction error to a minimum, in four variants (`rdp`, `grdp`,
  `rdp_fixed`, `mp_grdp`).
* **Post-processing**: 1-D clustering to merge nearby knees, filters that drop
  non-relevant ones, and ranking algorithms that score knee quality.
* **Deterministic by construction** *(new in 1.2.0)*: every rank and argmax in
  the library compares with a relative tolerance (`EPS_RANK`), so values that
  are mathematically equal but differ in their last bits are treated as tied and
  resolved by an explicit rule. Without this a knee could differ between two
  machines running identical input — see [Determinism](#determinism).
* **Shared curve primitives**: `utils` holds what more than one method needs —
  `detect_orientation` (which of the four knee/elbow shapes a curve is),
  `normalize` (both axes onto [0, 1]) and `span`. One implementation means one
  policy: a constant axis is handled the same way everywhere.

> **Note:** the library targets modern Python 3.12+ standards.

## Installation

```bash
pip install kneeliverse
```

From source:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install .
```

## Usage

Detecting a single knee:

```python
import numpy as np
import kneeliverse.lmethod as lmethod

# a cost curve: steep decline, then a flat tail
y = np.concatenate([np.linspace(1.0, 0.2, 10), np.full(30, 0.2)])
points = np.column_stack((np.arange(len(y), dtype=float), y))

knee = lmethod.knee(points)
print(f'knee at x={points[knee, 0]:.0f}')      # knee at x=9
```

Every detector shares that signature, so they are interchangeable —
`curvature.knee`, `dfdt.knee`, `kneedle.knee`, `lmethod.knee`, `menger.knee`.

`autoelbow` is the one that takes no parameters at all — the answer is a
property of the curve — and it handles all four orientations, so it does not
need to be told whether it is looking at a knee or an elbow:

```python
import kneeliverse.autoelbow as autoelbow
import kneeliverse.utils as utils

print(utils.detect_orientation(points))    # (decreasing, counter-clockwise)
print(autoelbow.knee(points))              # 9
```

`examples/compare_autoelbow.py` runs all six against each other on synthetic
and real curves, and reports where they disagree.

Multi-knee detection generalises *any* of them:

```python
import kneeliverse.multi_knee as mk

knees = mk.multi_knee(lmethod.knee, points)
```

On long or noisy curves, simplify first and map the result back. RDP cuts the
work the detector has to do without moving the answer:

```python
import kneeliverse.rdp as rdp

rng = np.random.default_rng(42)
x = np.arange(200, dtype=float)
y = np.exp(-x / 25.0) + rng.normal(0, 0.004, x.size)
points = np.column_stack((x, y))

reduced, removed = rdp.grdp(points, t=0.005)          # 200 points -> 178
idx = lmethod.knee(points[reduced])
knee = rdp.mapping(np.array([idx]), reduced, removed)[0]
print(f'knee at x={points[knee, 0]:.0f}')      # knee at x=5
```

## Determinism

Knee selection repeatedly takes a **discrete** decision — a rank, an argmax —
on **continuous** values. When two of those values are mathematically equal but
not bit-equal, an exact comparison turns last-bit arithmetic into a real
decision, and the answer starts depending on the platform's libm rather than on
the curve.

`knee_ranking.EPS_RANK` (1e-9, relative) states once how different two values
must be before the difference is allowed to matter, and the two primitives
built on it — `rank_min_tol` and `argmax_tol` — are used at every ranking and
selection site. Ties resolve to the leftmost candidate, the conservative knee.

Override the tolerance per call if your curve is not normalised to [0, 1]:

```python
import kneeliverse.knee_ranking as kr

scores = kr.right_flatness_ranking(points, knees, ratio_rtol=1e-6)
```

## Running unit tests

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install .
python -m unittest discover -s test
```

## Documentation

Documented with Google-style docstrings and published
[here](https://mariolpantunes.github.io/knee/). The docs are built and deployed
by `.github/workflows/docs.yml`; to preview them locally:

```bash
pip install pdoc
pdoc --math -d google -o docs_build kneeliverse \
  --logo "assets/logo.svg" --favicon "assets/logo.svg"
cp -r assets docs_build/assets
```

## Running the demos

```bash
python -m demos.curvature -i [trace]
python -m demos.dfdt -i [trace]
python -m demos.fusion -i [trace]
python -m demos.kneedle_classic -i [trace]
python -m demos.kneedle_rec -i [trace]
python -m demos.kneedle -i [trace]
python -m demos.lmethod -i [trace]
python -m demos.menger -i [trace]
python -m demos.zmethod -i [trace]
```

Most demos share the same options (`zmethod` and `kneedle_classic` differ):

```txt
usage: curvature.py [-h] -i I [-a] [-r R] [-t T] [-c C] [-o] [-g] [-k {left,linear,right,hull}]

Multi Knee evaluation app

options:
  -h, --help            show this help message and exit
  -i I                  input file
  -a                    add even spaced points
  -r R                  RDP reconstruction threshold
  -t T                  clustering threshold
  -c C                  corner threshold
  -o                    store output (debug)
  -g                    display output (debug)
  -k {left,linear,right,hull}
                        knee ranking method
```

## Authors

* [**Mário Antunes**](https://github.com/mariolpantunes)
* [**Tyler Estro**](https://www.fsl.cs.stonybrook.edu/~tyler/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Copyright

This project is under the following [COPYRIGHT](COPYRIGHT).
