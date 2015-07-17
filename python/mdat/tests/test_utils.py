

import numpy as np
import numpy.testing as npt
import mdat.utils as mdu

def test_form_D():
    npt.assert_equal(mdu.form_D(3), np.array([[-1, 1, 0],[0, -1, 1]]))
