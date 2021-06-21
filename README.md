# (Multi)Knee/Elbow point detection library

## Running unit tests

Several unit tests were written to validate some corner cases.
The unit tests were written in [unittest](https://docs.python.org/3/library/unittest.html).
Run the following commands to execute the unit tests.

```bash
python -m unittest
```

## Documentation

This library was documented using the google style docstring.
Run the following commands to the produce the documentation for this library.

```bash
pip install pdoc3
pdoc -c latex_math=True --html -o docs knee --force
```

## Instalation



![Python CI](https://github.com/mariolpantunes/knee/workflows/Python%20CI/badge.svg)