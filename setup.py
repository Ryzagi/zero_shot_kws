from pathlib import Path
from typing import List

from setuptools import find_packages, setup

THIS_DIR = Path(__file__).parent


def _load_requirements(path_dir: Path, comment_char: str = "#") -> List[str]:
    """Load the requirements from the specified file.

    Args:
        path_dir: The path to the directory containing the requirements file.
        comment_char: The character used to denote comments in the requirements file.

    Returns:
        A list of the requirements in the file.
    """
    requirements_directory = path_dir / "requirements.txt"
    requirements = []
    with requirements_directory.open("r") as file:
        for line in file.readlines():
            # Strip leading whitespace from the line
            line = line.lstrip()

            # Remove comments from the line
            if comment_char in line:
                line = line[: line.index(comment_char)]

            # Add the requirement to the list if it is not an empty line
            if line:
                requirements.append(line)

    return requirements



setup(
    name="zerospot",
    version="0.0.1",
    description="TODO",
    packages=find_packages(exclude=["tests", "tests.*"]),
    install_requires=_load_requirements(THIS_DIR),
)
