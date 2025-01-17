import numpy as np
from nose.tools import assert_raises
from numpy.testing import assert_allclose

from genz.static.expyfun import repeated_mls, compute_mls_impulse_response


def test_mls_ir():
    """Test computing impulse response with MLS
    """
    # test simple stuff
    for _ in range(5):
        # make sure our signals have some DC
        sig_len = np.random.randint(10, 2000)
        kernel = np.random.rand(sig_len) + 10 * np.random.rand(1)
        n_repeats = 10

        mls, n_resp = repeated_mls(len(kernel), n_repeats)
        resp = np.zeros(n_resp)
        resp[:len(mls) + len(kernel) - 1] = np.convolve(mls, kernel)

        est_kernel = compute_mls_impulse_response(resp, mls, n_repeats)
        kernel_pad = np.zeros(len(est_kernel))
        kernel_pad[:len(kernel)] = kernel
        assert_allclose(kernel_pad, est_kernel, atol=1e-5, rtol=1e-5)

    # failure modes
    assert_raises(TypeError, repeated_mls, 'foo', n_repeats)
    assert_raises(ValueError, compute_mls_impulse_response, resp[:-1], mls,
                  n_repeats)
    assert_raises(ValueError, compute_mls_impulse_response, resp, mls[:-1],
                  n_repeats)
    assert_raises(ValueError, compute_mls_impulse_response, resp,
                  mls * 2. - 1., n_repeats)
    assert_raises(ValueError, compute_mls_impulse_response, resp,
                  mls[np.newaxis, :], n_repeats)
