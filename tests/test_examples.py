#!/usr/bin/env python

from pathlib import Path

import matplotlib
import pytest

matplotlib.interactive(False)


here = Path(__file__).parent.resolve()
example_folder = here.parent / 'examples'
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
    with open(path) as f:
        code = compile(f.read(), name, 'exec')
        exec(code, {})


