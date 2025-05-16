#!/usr/bin/env python

import subprocess
from pathlib import Path

import pytest

here = Path(__file__).parent.resolve()
example_folder = here.parent / "examples"
examples = example_folder.glob("*.py")


def generate_examples():
    for example in examples:
        test_name = example.stem
        file_path = example_folder / example
        yield file_path, test_name


@pytest.mark.parametrize("path, name", list(generate_examples()))
def test_run(path, name):
    """
    Check that the examples can be executed.
    """
    # subprocess.run(['python', '-c', "import matplotlib; matplotlib.interactive(False); exec(open('{}').read())".format(path)], check=True)
    subprocess.run(["python", "-c", f"exec(open('{path}').read())"], check=True)
