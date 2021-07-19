import logging
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pkg_resources
from matplotlib import cycler

logger = logging.getLogger(__name__)


COLOR_ORIG = '#FF4D5B'
COLOR_SYNTH = '#312874'


# -- Plotting functions
def set_plotting_style():
    plt.style.use('seaborn')
    font_file = "fonts/inter-v3-latin-regular.ttf"
    try:
        mpl.font_manager.fontManager.addfont(
            Path(pkg_resources.resource_filename("synthesized", font_file)).as_posix()
        )
        mpl.rc('font', family='Inter-Regular')
    except FileNotFoundError:
        warnings.warn(f"Unable to load '{font_file}'")

    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['text.color'] = '333333'
    mpl.rcParams['font.family'] = 'inter'
    mpl.rcParams['axes.facecolor'] = 'EFF3FF'
    mpl.rcParams['axes.edgecolor'] = '333333'
    mpl.rcParams['axes.grid'] = True
    mpl.rcParams['grid.color'] = 'D7E0FE'
    mpl.rcParams['xtick.direction'] = 'out'
    mpl.rcParams['ytick.direction'] = 'out'
    mpl.rcParams['axes.prop_cycle'] = cycler('color', ['312874', 'FF4D5B', 'FFBDD1', '4EC7BD', '564E9C'] * 10)
