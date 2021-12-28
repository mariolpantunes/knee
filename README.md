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

The library can be used by adding this line to the requirement.txt file:
```txt
git+git://github.com/mariolpantunes/knee@main#egg=knee
```

## Runing the demos

The demos can be execute as python modules using the following code:

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
Most demos have the same parameters (with the exception of zmethod and kneedle_classic):

```bash
python -m demos.curvature -husage: curvature.py [-h] -i I [-a] [-r R] [-t T] [-c C] [-o] [-g] [-k {left,linear,right,hull}]

Multi Knee evaluation app

optional arguments:
  -h, --help            show this help message and exit
  -i I                  input file
  -a                    add even spaced points
  -r R                  RDP reconstruction threshold
  -t T                  clustering threshold
  -c C                  corner threshold
  -o                    store output (debug)
  -g                    display output (debug)
  -k {left,linear,right,hull}
                        Knee ranking method
```

![Python CI](https://github.com/mariolpantunes/knee/workflows/Python%20CI/badge.svg)