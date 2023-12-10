# (Multi)Knee/Elbow point detection library

Estimating the knee/elbow point in performance curves is a challenging task.
However, most of the time these points represent ideal compromises between cost and performance.

This library implements several well-known knee detection algorithms:
1. Discrete Curvature 
2. DFDT
3. Kneedle
4. L-method
5. Menger curvature

Furthermore, the code in this library expands the ideas on these algorithms to 
detect multi-knee/elbow points in complex curves.
We implemented a recursive method that allows each of the previously mentioned methods
to detect multi-knee and elbow points.
Some methods natively support multi-knee detection, such as:
1. Kneedle
2. Fusion
3. Z-method

Finally, we also implemented additional methods that help with knee detection tasks.
As a preprocessing step, we develop a custom RDP algorithm that reduced a discrete 
set of points while keeping the reconstruction error to a minimum.
As a post-processing step we implemented several algorithms:
1. 1D dimensional clustering, is used to merge close knee points
2. Several filters out non-relevant knees
3. Knee ranking algorithms that used several criteria to assess the quality of a knee point

## Running unit tests

Several unit tests were written to validate some corner cases.
The unit tests were written in [unittest](https://docs.python.org/3/library/unittest.html).
Run the following commands to execute the unit tests.

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install .
python -m unittest
```

## Documentation

This library was documented using the google style docstring, it can be accessed [here](https://mariolpantunes.github.io/knee/).
Run the following commands to produce the documentation for this library.

```bash
python3 -m venv venv
source venv/bin/activate
pip install pdoc
pdoc --math -d google -o docs src/knee \
--logo https://raw.githubusercontent.com/mariolpantunes/knee/main/media/knee.png \
--favicon https://raw.githubusercontent.com/mariolpantunes/knee/main/media/knee.png
```

## Instalation

To install the library locally, simple execute the following commands:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install .
```
You can also use the PyPI repository for easy access to the library:

```txt
knee>=0.1
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

## Authors

* [**Mário Antunes**](https://github.com/mariolpantunes)

* [**Tyler Estro**](https://www.fsl.cs.stonybrook.edu/~tyler/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Copyright

This project is under the following [COPYRIGHT](COPYRIGHT).

![Python CI](https://github.com/mariolpantunes/knee/workflows/Python%20CI/badge.svg)
