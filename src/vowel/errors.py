"""Custom exception types for the vowel evaluation framework."""


class SignatureError(Exception):
    """Raised when a function signature doesn't comply with fixture requirements.

    This is raised when:
    - A fixture parameter is positional instead of keyword-only (after *)
    - A fixture parameter is missing from the function signature
    - The function uses *args or **kwargs (non-deterministic)
    - A teardown function has more than one required parameter

    Examples:
        * Valid:
            - def fn(x: int, *, db: Any): ...

        * Raises SignatureError:
            - def fn(x: int, db: Any): ...   # db must be keyword-only
    """

    pass


class FixturePathError(Exception):
    """Raised when a fixture import path points to the calling module (__main__).

    Importing the calling module would re-execute the script and cause a nested
    evaluation run (RuntimeError: A task run has already been entered).
    """

    pass
