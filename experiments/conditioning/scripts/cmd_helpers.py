"""CMD reference curves extracted from the paper plotting code."""
import numpy as np
H1_TO_H70 = -0.7745

def masters10_red_edge(mr: np.ndarray) -> np.ndarray:
    """Masters et al. (2010) eq. (1), the *blue edge of the red sequence*.

    Galaxy Zoo: Passive Red Spirals, MNRAS 411, 2026 (arXiv:0910.4113). Note what it
    is: they fit the red sequence of early types (p_el >= 0.8) as
    (g-r) = 0.73 - 0.02 (M_r + 20) with scatter sigma = 0.1, then place this line one
    sigma blueward of that ridge to select red spirals. So it is a conservative red
    cut, not a valley-following divider like Baldry's, and it should be expected to
    sit below the density minimum rather than in it.

    The slope is shallow enough that the distance scale barely matters: the 0.7745 mag
    between h = 1 and h70 moves the line by 0.015 mag in colour, well under the width
    of the green valley. Contrast the Baldry tanh, where the same shift moved the
    divider by up to 0.12 mag and had to be corrected.
    """
    return 0.63 - 0.02 * (mr + 20.0)

def masters10_ridge(mr: np.ndarray) -> np.ndarray:
    """The red-sequence ridge that cut is defined relative to."""
    return 0.73 - 0.02 * (mr + 20.0)

