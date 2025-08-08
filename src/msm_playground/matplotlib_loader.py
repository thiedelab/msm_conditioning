import os
import matplotlib as mpl
from cycler import cycler
from enum import Enum

# Automatically locate the repository root and the .matplotlibrc file
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Root of the repo
rcfile_path = os.path.join(repo_root, '.matplotlibrc')

class CycleStyle(Enum):
    MARKERS = 'markers'
    LINES = 'lines'

# Some presets I had issue specifying in the .matplotlibrc file with
def set_hardcoded_presets(cycle_style: CycleStyle = CycleStyle.MARKERS):
    if(cycle_style == CycleStyle.MARKERS):
        custom_cycler = cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22'])  + cycler('marker', ['o', 's', '^', 'D', 'v', '>', '<', 'p', 'h'])
    elif(cycle_style == CycleStyle.LINES):
        custom_cycler = cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])  + cycler('linestyle', ['-', '--', '-.', ':'])
    mpl.pyplot.rc('axes', prop_cycle=custom_cycler)

def fix_axis_intersection_at_y0(draw_grid: bool=True):
    """
    This function ensures that the y-axis intersects the x-axis at Y=0
    and hides the top and right spines for a cleaner look.
    """
    ax = mpl.pyplot.gca()  # Get the current axis
    
    # Ensure that the bottom spine (x-axis) is always at y=0
    ax.spines['bottom'].set_position(('data', 0))
    
    # Optionally, hide the top and right spines for a cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Draw grid
    if draw_grid:
        ax.grid(True, which='both', axis='both', linewidth=0.5)

def load_matplotlib_presets(cycle_style: CycleStyle = CycleStyle.MARKERS):
    if os.path.exists(rcfile_path):
        mpl.rc_file(rcfile_path)
        print(f"Loaded .matplotlibrc from {rcfile_path}")
    else:
        print(f".matplotlibrc file not found at {rcfile_path}, using default settings.")
    set_hardcoded_presets(cycle_style)
