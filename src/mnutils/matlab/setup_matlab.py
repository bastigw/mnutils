import os

import matlab.engine
from loguru import logger


def connect_to_matlab() -> matlab.engine.MatlabEngine:
    """Connect to an existing shared MATLAB session, or start a new one.

    Make sure to only call this function once per session — a second call
    to connect to the same shared engine raises an `EngineError`.

    Returns
    -------
    matlab.engine.MatlabEngine
        The connected (or newly started) MATLAB engine instance.
    """
    engines = matlab.engine.find_matlab()
    if engines:
        logger.debug(f"Found running shared MATLAB engines. Connecting to engine {engines[0]}")
        try:
            eng = matlab.engine.connect_matlab(engines[0])
        except matlab.engine.EngineError as e:  # type: ignore
            logger.error(f"EngineError while connecting to MATLAB engine {engines[0]}: {e}")
            logger.error("This may be due to multiple connections. Only one connection is allowed.")
            raise
        except (RuntimeError, TypeError, OSError) as e:
            logger.error(f"Failed to connect to MATLAB engine {engines[0]}: {e}")
            raise
    else:
        logger.debug(
            "No running MATLAB engine found. Starting a new MATLAB engine. "
            "This may take a second..."
        )
        eng = matlab.engine.start_matlab()

    if not isinstance(eng, matlab.engine.MatlabEngine):
        raise TypeError(f"Expected eng to be of type matlab.engine.MatlabEngine, got {type(eng)}")
    logger.debug("Connected to MATLAB engine successfully.")
    return eng


def setup_util_path(
    eng: matlab.engine.MatlabEngine,
    matlab_utils_path: str | None = os.environ.get("MATLAB_UTILS"),
) -> None:
    """Add the MATLAB utils path to the MATLAB engine's path, if not already present.

    Parameters
    ----------
    eng : matlab.engine.MatlabEngine
        MATLAB engine instance.
    matlab_utils_path : str | None
        Path to the MATLAB utils folder. Defaults to the `MATLAB_UTILS` environment
        variable.
    """
    if matlab_utils_path and os.path.exists(matlab_utils_path):
        logger.debug(f"Adding MATLAB utils path: {matlab_utils_path}")
        currentMatlabPath = str(eng.path())
        if matlab_utils_path not in currentMatlabPath.split(":"):
            eng.addpath(matlab_utils_path)
        else:
            logger.debug(f"MATLAB utils path already exists: {matlab_utils_path}")
    else:
        logger.warning(f"MATLAB_UTILS path is not set or does not exist: {matlab_utils_path}")


def add_matlablatest_path(eng: matlab.engine.MatlabEngine) -> None:
    """Add the 'matlablatest' path to the MATLAB environment using `manage_paths`.

    Parameters
    ----------
    eng : matlab.engine.MatlabEngine
        MATLAB engine instance.

    Raises
    ------
    ValueError
        If the MATLAB engine instance is invalid, or no 'matlabfiles' paths
        are found after adding 'matlablatest'.
    RuntimeError
        If the 'manage_paths' function is not available in MATLAB.
    """
    try:
        if not isinstance(eng, matlab.engine.MatlabEngine):
            raise ValueError("Invalid MATLAB engine instance provided.")

        logger.debug("Attempting to add 'matlablatest' path using manage_paths...")

        # Check if the 'manage_paths' function exists in MATLAB
        if not eng.which("manage_paths"):
            raise RuntimeError("'manage_paths' function not found in MATLAB environment.")

        # Add the 'matlablatest' path
        eng.manage_paths("matlablatest", True, nargout=0)

        # Retrieve and log paths containing 'matlabfiles'
        matlabPath = str(eng.path())
        matlabfiles_paths = [p for p in matlabPath.split(":") if "matlabfiles" in p.lower()]
        if not matlabfiles_paths:
            matlab_sources = eng.eval("getenv('MATLAB_SOURCES')", nargout=1)
            raise ValueError(
                f"No 'matlabfiles' paths found after adding 'matlablatest'. "
                f"Check if MATLAB_SOURCES is set correctly. Current value: {matlab_sources}"
            )
        logger.debug("Successfully added 'matlablatest' path.")

    except ValueError as ve:
        logger.error(f"ValueError: {ve}")
        raise
    except RuntimeError as re:
        logger.error(f"RuntimeError: {re}")
        raise
    except (AttributeError, TypeError) as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise
