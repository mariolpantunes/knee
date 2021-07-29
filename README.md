# (Multi)Knee/Elbow point detection library

## Running unit tests

Several unit tests were written to validate some corner cases.
The unit tests were written in [unittest](https://docs.python.org/3/library/unittest.html).
Run the following commands to execute the unit tests.

```bash
python -m unittest
```

## Documentation

This library was documented using the google style docstring, it can be accessed [here](https://mariolpantunes.github.io/knee/).
Run the following commands to produce the documentation for this library.

```bash
pip install pdoc3
pdoc -c latex_math=True --html -o docs knee --force
```

## Instalation

## Runing the demos

The demos can be execute as python modules using the following code:

```bash
python -m demos.curvature -i [trace]
python -m demos.dfdt -i [trace]
python -m demos.kneedle -i [trace]
python -m demos.kneedle_rec -i [trace]
python -m demos.lmethod -i [trace]
python -m demos.menger -i [trace]
python -m demos.rdp -i [trace]
python -m demos.zscore -i [trace]
```
Most demos have the same parameters (with the exception of zscore):

```bash
python -m demos.curvature -h
usage: curvature.py [-h] -i I [-r R] [-c {single,complete,average}] [-t T] [-m {left,linear,right}] [-o] [-a | -b]

Multi Knee evaluation app

optional arguments:
  -h, --help            show this help message and exit
  -i I                  input file
  -r R                  RDP R2
  -c {single,complete,average}
                        clustering metric
  -t T                  clustering threshold
  -m {left,linear,right}
                        direction of the cluster ranking
  -o                    store output (debug)
  -a                    add even spaced points (rdp based)
  -b                    add even spaced points (knee based)
```

![Python CI](https://github.com/mariolpantunes/knee/workflows/Python%20CI/badge.svg)