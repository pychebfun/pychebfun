"""
pychebfun Tests

Used by setup.py to automatically run all pychebfun tests.
"""


def test(dir='pychebfun'):
    """
    Run all tests scripts under directories named `test` with `dir` as
    the top directory. (Top pychebfun directory by default.)
    """
    test_files = get_test_files(dir)



def get_test_files(dir):
    """
    Returns all files under all `dir` subdirectories named `test`.
    """
    
