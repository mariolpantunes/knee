# Changelog

## 1.2.0

### Added

- **`autoelbow`** — the AutoElbow method (Onumanyi et al., *Applied Sciences*
  12(15):7515, 2022). Scores each point by a ratio of squared distances to
  three references and takes the largest. It is the only detector here with no
  parameters at all, and it handles all four knee/elbow orientations.
- **`utils`** — the primitives more than one method needs:
  - `detect_orientation` — which of the four shapes a curve is. Extracted from
    `kneedle`, where it was inlined and untested.
  - `normalize` — both axes onto [0, 1], with one policy for a constant axis.
  - `span` — the extent of each axis.
  - `Direction` and `Concavity` moved here from `kneedle`, which re-exports
    them, so `kneedle.Direction` keeps working.
- **`knee_ranking.argmax_tol`** and **`rank_min_tol`** — selection and ranking
  that treat values equal within a relative tolerance as tied, with
  **`EPS_RANK`** (1e-9) as the library's single statement of "how different is
  different".
- **`evaluation.compute_metric`** — dispatches the `Metrics` enum to the
  function that implements it. Nothing mapped one to the other before, so
  every caller wrote the branch by hand.
- Examples: `compare_autoelbow.py`, `triangle_area_corners.py`,
  `rdp_cost_and_frames.py`.
- `plot` optional extra for `rdp.plot_frame`'s matplotlib dependency.

### Fixed

- **`menger.knee` returned index 1 for every curve.** The `fabs` in the
  curvature numerator covered only its first term, so collinear points scored
  1.06 instead of 0, a straight run scored uniformly non-zero, and the whole
  run tied. It now agrees with `curvature.knee` on 5 of 5 traces, up from 2.
- **Knee selection was not reproducible across machines.** Ranks and argmaxes
  compared floats exactly, so values that were mathematically equal but not
  bit-equal were separated by last-bit arithmetic — the same input picked a
  different knee on two different machines. Every ranking and selection site
  now uses the tolerant primitives.
- **`slope_ranking`** ranked with `rank`, which splits ties into consecutive
  integers in sort order. On a linear decline it returned `[0, 0.667, 0.333, 1]`
  and picked the rightmost knee.
- **`rank_corners_triangle`** inlined `0.5*(x1-x0)*(y1-y2)` behind a
  `TODO: use the above function`. That is a one-sided step, not a bend, and it
  was signed, so a knee in a rising run could never win its cluster. It now
  uses `triangle_area`; on the traces, all 5 changed selections move to a
  sharper corner.
- **`triangle_area`** returned the signed shoelace determinant while being
  documented as an area, so it flipped sign with the winding order.
- **`evaluation.compute_global_segment_cost`** raised on every call — it
  unpacked a single return value as a pair, then called `compute_cost` with
  three of its four arguments, and the wrong ones.
- **`rdp.plot_frame`** raised `NameError` on every call, referencing `plt`
  from a commented-out import. matplotlib is now imported lazily and the
  output directory is a parameter.
- **`evaluation.get_neighbourhood`** raised `NameError` for `t >= 1.0`.
- **`rdp.min_point_rdp`** sorted the caller's threshold list in place, and its
  mutable default made that persist between calls.
- **`kneedle.knees`** divided by zero on a curve with a constant axis;
  `kneedle.knee`, in the same module, guarded it. Both now share `normalize`.
- `np.cross` on 2-D vectors, deprecated in NumPy 2.0 and slated for removal,
  replaced with the module's own `cross2d`.
- Annotation corrections where the declared type was not what the function
  returned: `lmethod.get_knee` (`int` vs a 3-tuple), `kneedle.knee` (`int` vs
  `int|None`), and seven parameters annotated `callable` (the builtin
  predicate) rather than `Callable`.

### Changed

- **Python 3.12+** is required (was 3.8).
- Packaging moved to PEP 621 `pyproject.toml`; `setup.cfg` removed.
- CI now gates on ruff, basedpyright and vulture across Python 3.12 and 3.14,
  matching the `ess`, `torann` and `pyBlindOpt` repositories.
- Documentation is built and published by `.github/workflows/docs.yml`;
  generated HTML is no longer committed.
- New `assets/logo.svg`, replacing `media/knee.png`.
- Test coverage of the public API went from 59% to 94% — 106 tests to 286.

### Removed

- Nothing. Two functions were deleted mid-development and restored, working,
  before release.

## 1.0.1 and earlier

See the commit history.
