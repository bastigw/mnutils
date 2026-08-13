# MNUtils

MNUtils is a collection of utility functions for working with multi-nuclear datasets from our GE scanners.

## Features

- Simplifies handling and processing of multi-nuclear data

## Installation

```bash
uv add git+https://gitlab.developers.cam.ac.uk/deptofrad/molecular-imaging/multi-nuclear-utils-python.git
```

## Usage

```python
from mnutils import get_mat_data_from_series

# Example: Get raw data path from a specific series in a dataset
BASE_PATH = "data/HeVo-11"

raw_data_path = get_mat_data_from_series(BASE_PATH, 29)
print(raw_data_path)
```

## Requirements

- Python 3.12+

## MATLAB Engine Dependency

To use MNUtils, you need to install the MATLAB Engine for Python. The version of the MATLAB Engine you require depends on your local MATLAB version. Ensure you install the correct version that matches your MATLAB installation.

You can find the appropriate MATLAB Engine version for your setup here: [MATLAB Engine for Python](https://pypi.org/project/matlabengine/#history).

To install the MATLAB Engine in your project, use the following command, replacing `<version>` with the correct version for your MATLAB:

```bash
uv add "matlabengine==<version>"
```

## Contributions and Suggestions

I'm very open to suggestions and improvements! If you have ideas or want to share your own utilities, feel free to reach out or submit a pull request.

## Licence

These scripts are free to use, modify, and share. I hope they help you in your projects! (For now at least internally)
