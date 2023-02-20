# DPGen

Proof-of-Concept automated program synthesizer for differential privacy. Code for [CCS'21] Automated Program Synthesis
for Differential Privacy.

# Setup

We provide a docker container for experiments, use `docker pull ghcr.io/cmla-psu/dpgen` to pull the image
or `docker build . -t dpgen`
to build the image yourself. Then, run `docker run --rm --it dpgen` and you will be inside `dpgen` root folder with
everything installed.

However, we highly recommend installing `dpgen` in a `conda` environment for best performance:

```bash
# The minimum supported python version is 3.9.
conda create -n dpgen anaconda python=3.9
conda actiavte dpgen

# install dependencies from conda for best performance
conda install numpy numba sympy tqdm coloredlogs pip
# install icc_rt compiler for best performance with numba, this requires using intel's channel
conda install -c intel icc_rt
# install the remaining non-conda dependencies and dpgen 
pip install .
```

# Usage

We use a subset of python syntax as the frontend of `dpgen` (i.e., the DSL of `dpgen` is disguised as a subset of
python). You can directly write a non-private python function and pass the function to the top-level API
`dpgen.privatize` to automatically generate a differentially-private version of it.

To give a closer look at how `dpgen` works in action, take Sparse Vector for example:

```python
import ast

from dpgen.frontend.annotation import output, ListBound
from dpgen import privatize


# first, define your non-private function that you'd like to privatize, remember to use type hints to properly annotate
# the types of the input variables.
# Note that currently only a small subset of python syntax is supported, the frontend compiler will raise exceptions if 
# a non-supported syntax is used.
def sparse_vector(q: list[float], size: int, t: float, n: int):
    i, count = 0, 0
    while i < size and count < n:
        if q[i] >= t:
            output(True)
            count += 1
        else:
            output(False)
        i = i + 1


# Then, you can call `dpgen.privatize` function to automatically synthesize a differentially-private mechanism (
# in `ast.AST` form).
tree = privatize(
    sparse_vector,  # the function variable
    privates={'q'},  # annotate private input variables
    # constraint function for the parameters, here one of the precondition is `n < size / 5`. Note that the constraint
    # function _must_ take the same parameters as the target function.
    constraint=lambda q, size, t, n: n < size / 5,
    # specify bounds for the input variables to constrain the search space, the key must be the name of an input variable
    # and the value can be a function from input variables to (left_bound, right_bound) or simply a (left_bound, right_bound).
    # Here, we specify the bounds for input variable `n` to be within the range of (0, size). 
    original_bounds={'n': lambda q, size, t, n: (0, size)},
    # We should also specify the bounds on the related execution (i.e., the hat variables). For convenience, you can use
    # dpgen.frontend.annotation.ListBound.ALL_DIFFER or ONE_DIFFER to determine all or only one element(s) in the list 
    # in the related execution can differ.
    related_bounds={'q': ListBound.ALL_DIFFER}
)

# Finally, you can run `ast.unparse` to convert the AST form back to source code.
code = ast.unparse(tree)
print(code)
```

Note that we also provide a `benchmark.py` for running `dpgen` all algorithms studied in the paper.

# Citation

Please consider citing the following paper if you find this tool useful in your academic paper:

```bibtex
@inproceedings{10.1145/3460120.3484781,
  author = {Wang, Yuxin and Ding, Zeyu and Xiao, Yingtai and Kifer, Daniel and Zhang, Danfeng},
  title = {DPGen: Automated Program Synthesis for Differential Privacy},
  year = {2021},
  isbn = {9781450384544},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3460120.3484781},
  doi = {10.1145/3460120.3484781},
  booktitle = {Proceedings of the 2021 ACM SIGSAC Conference on Computer and Communications Security},
  pages = {393–411},
  numpages = {19},
  keywords = {program synthesis, differential privacy},
  location = {Virtual Event, Republic of Korea},
  series = {CCS '21}
}
```

