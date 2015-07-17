"""Testing uility functions """

import numpy as np
import numpy.testing as npt
import mdat.utils as mdu


def test_form_D():
    npt.assert_equal(mdu.form_D(3), np.array([[-1, 1, 0],[0, -1, 1]]))

def test_form_DxInv():
    npt.assert_raises(ValueError, mdu.form_DxInv, np.array([1,3,2]), 2)
    x = np.array([1,2,3,4,5,6])
    npt.assert_equal(mdu.form_DxInv(x, 2), np.diag([0.5, 0.5, 0.5, 0.5]))

def test_combine_by_x():

    x = np.array([1, 1, 2, 2 + 1e-15, 3])
    Y = np.array([[1, 2], [2, 2], [2, 2], [2, 2], [2, 2]])
    #[1.5, 2.5, 3]
    new_Y, weights, unique_x = mdu.combine_by_x(x, Y)
    npt.assert_array_equal(new_Y, np.array([[1.5, 2], [2, 2], [2,2]]))
