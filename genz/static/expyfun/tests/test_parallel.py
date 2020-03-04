# -*- coding: utf-8 -*-

import warnings

import numpy as np
from nose.tools import assert_raises
from numpy.testing import assert_array_equal

from genz.static.expyfun import parallel_func, _check_n_jobs
from genz.static.expyfun import requires_lib

warnings.simplefilter('always')


def fun(x):
    return x


@requires_lib('joblib')
def test_parallel():
    """Test parallel support."""
    assert_raises(TypeError, _check_n_jobs, 'foo')
    parallel, p_fun, _ = parallel_func(fun, 1)
    a = np.array(parallel(p_fun(x) for x in range(10)))
    parallel, p_fun, _ = parallel_func(fun, 2)
    b = np.array(parallel(p_fun(x) for x in range(10)))
    assert_array_equal(a, b)
