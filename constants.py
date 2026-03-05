"""
Global constants and configuration
No external module imports - this is the foundation
"""

# Numerical tolerance
TOLERANCE = 1e-10

# Maximum orbital angular momentum quantum number
MAX_ORBITAL_L = 12
MIN_POWER_THRESHOLD = 10

# Category ordering for display
CATEGORY_ORDER = [
    'Nonaxial groups',
    'Cn groups',
    'Dn groups',
    'Cnv groups',
    'Cnh groups',
    'Dnh groups',
    'Dnd groups',
    'Sn groups',
    'Cubic groups'
]

# Mathematical constants
SQRT2 = 2 ** 0.5
SQRT3 = 3 ** 0.5
SQRT5 = 5 ** 0.5

# Version and metadata
__version__ = "2.0.0"
__author__ = "Jianwen Ma"
__license__ = "MIT"