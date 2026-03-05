"""Character table data storage and loading"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import json
import cmath
import math

# Import from our modules
from constants import CATEGORY_ORDER, TOLERANCE

@dataclass
class CharacterTable:
    """Immutable data structure for a character table"""
    name: str
    irreps: Dict[str, List[complex]]
    class_sizes: List[int]
    class_names: List[str]
    order: int
    category: str
    vector_char: Optional[List[complex]] = None
    class_cycles: Optional[Dict] = None
    special_notes: Optional[str] = None
    is_complex: bool = False
    
    @classmethod
    def from_dict(cls, name: str, data: Dict) -> 'CharacterTable':
        """Create from dictionary"""
        return cls(
            name=name,
            irreps=data['irreps'],
            class_sizes=data['class_sizes'],
            class_names=data['class_names'],
            order=data['order'],
            category=data['category'],
            vector_char=data.get('vector_char'),
            class_cycles=data.get('class_cycles'),
            special_notes=data.get('special_notes'),
            is_complex=data.get('complex', False)
        )

class CharacterTableDatabase:
    """Manages character table data"""
    
    def __init__(self):
        self.tables: Dict[str, CharacterTable] = {}
        self._init_all_tables()
    
    def _init_all_tables(self):
        """Initialize all point groups"""
        # Pre-computed constants
        omega = cmath.exp(2j * cmath.pi / 3)
        omega_squared = cmath.exp(4j * cmath.pi / 3)
        i = cmath.exp(1j * cmath.pi / 2)                     # exp(iπ/2)
        epsilon5 = cmath.exp(2j * cmath.pi / 5)              # exp(2πi/5)
        epsilon10 = cmath.exp(2j * cmath.pi / 10)              # exp(2πi/10)
        epsilon6 = cmath.exp(2j * cmath.pi / 6)              # exp(πi/3)
        epsilon6_star = epsilon6.conjugate()
        eta_plus = (1 + math.sqrt(5)) / 2
        eta_minus = (1 - math.sqrt(5)) / 2
        cos72 = math.cos(2 * math.pi / 5)
        cos144 = math.cos(4 * math.pi / 5)
        two_cos72 = 2 * cos72
        two_cos144 = 2 * cos144
        sqrt2 = math.sqrt(2)
        sqrt3 = math.sqrt(3)
        # -----------------------------------------------------
        
        # Define tables (extracted from original code)
        tables_data = {
            # ---- Nonaxial groups ----
            # C₁ 群
            'C_1': {
                'irreps': {'A': [1]},
                'class_sizes': [1],
                'class_names': ['E'],
                'order': 1,
                'class_cycles': {0: (1, [0])},
                'vector_char': [3],
                'category': 'Nonaxial groups'
            },

            # Cₛ 群（又名 C₁h）
            'C_s': {
                'irreps': {
                    'A\'':  [1, 1],
                    'A\'\'': [1, -1]
                },
                'class_sizes': [1, 1],
                'class_names': ['E', 'σh'],
                'order': 2,
                'class_cycles': {
                    0: (1, [0]),
                    1: (2, [0, 1])
                },
                'vector_char': [3, 1],
                'category': 'Nonaxial groups'
            },

            # ---- Cn groups ----
            # C₂ 群
            'C_2': {
                'irreps': {
                    'A': [1, 1],
                    'B': [1, -1]
                },
                'class_sizes': [1, 1],
                'class_names': ['E', 'C₂'],
                'order': 2,
                'class_cycles': {
                    0: (1, [0]),
                    1: (2, [0, 1])
                },
                'vector_char': [3, -1],
                'category': 'Cn groups'
            },

            # C₃ 群
            'C_3': {
                'irreps': {
                    'A': [1, 1, 1],
                    'E': [1, omega, omega_squared],
                    'E\'': [1, omega_squared, omega]
                },
                'class_sizes': [1, 1, 1],
                'class_names': ['E', 'C₃', 'C₃²'],
                'order': 3,
                'class_cycles': {
                    0: (1, [0]),           # E
                    1: (3, [0, 1, 2]),     # C₃
                    2: (3, [0, 2, 1])      # C₃²
                },
                'vector_char': [3, 0, 0],
                'category': 'Cn groups',
                'complex': True,
                'special_notes': 'ω = exp(2πi/3), ω² = exp(4πi/3)'
            },

            # C₄ 群
            'C_4': {
                'irreps': {
                    'A': [1, 1, 1, 1],
                    'B': [1, -1, 1, -1],
                    'E1': [1, 1j, -1, -1j],
                    'E2': [1, -1j, -1, 1j]
                },
                'class_sizes': [1, 1, 1, 1],
                'class_names': ['E', 'C₄', 'C₂', 'C₄³'],
                'order': 4,
                'class_cycles': {
                    0: (1, [0]),           # E
                    1: (4, [0, 1, 2, 3]), # C₄
                    2: (2, [0, 2]),       # C₂
                    3: (4, [0, 3, 2, 1])  # C₄³
                },
                'vector_char': [3, 1, -1, 1],
                'category': 'Cn groups',
                'complex': True,
                'special_notes': 'i = exp(πi/2)'
            },

            # C₅ 群
            'C_5': {
                'irreps': {
                    'A':   [1, 1, 1, 1, 1],
                    'E1':  [1, epsilon5, epsilon5**2, epsilon5**3, epsilon5**4],
                    'E1*': [1, epsilon5**4, epsilon5**3, epsilon5**2, epsilon5],
                    'E2':  [1, epsilon5**2, epsilon5**4, epsilon5, epsilon5**3],
                    'E2*': [1, epsilon5**3, epsilon5, epsilon5**4, epsilon5**2]
                },
                'class_sizes': [1, 1, 1, 1, 1],
                'class_names': ['E', 'C₅', 'C₅²', 'C₅³', 'C₅⁴'],
                'order': 5,
                'class_cycles': {
                    0: (1, [0]),               # E
                    1: (5, [0, 1, 2, 3, 4]),   # C₅
                    2: (5, [0, 2, 4, 1, 3]),   # C₅²
                    3: (5, [0, 3, 1, 4, 2]),   # C₅³
                    4: (5, [0, 4, 3, 2, 1])    # C₅⁴
                },
                'vector_char': [3, eta_plus, eta_minus, eta_minus, eta_plus],
                'category': 'Cn groups',
                'complex': True,
                'special_notes': 'ε = exp(2πi/5), η⁺ = 1+2cos72° ≈ 1.618, η⁻ = 1+2cos144° ≈ -0.618'
            },

            # C₆ 群
            'C_6': {
                'irreps': {
                    'A':   [1,  1,  1,  1,  1,  1],
                    'B':   [1, -1,  1, -1,  1, -1],
                    'E1':  [1,  epsilon6, -epsilon6_star, -1, -epsilon6,  epsilon6_star],
                    'E1*': [1,  epsilon6_star, -epsilon6, -1, -epsilon6_star,  epsilon6],
                    'E2':  [1, -epsilon6_star, -epsilon6,  1, -epsilon6_star, -epsilon6],
                    'E2*': [1, -epsilon6, -epsilon6_star,  1, -epsilon6, -epsilon6_star]
                },
                'class_sizes': [1, 1, 1, 1, 1, 1],
                'class_names': ['E', 'C₆', 'C₃', 'C₂', 'C₃²', 'C₆⁵'],
                'order': 6,
                'class_cycles': {
                    0: (1, [0]),               # E
                    1: (6, [0, 1, 2, 3, 4, 5]), # C₆
                    2: (3, [0, 2, 4]),         # C₃
                    3: (2, [0, 3]),            # C₂
                    4: (3, [0, 4, 2]),         # C₃²
                    5: (6, [0, 5, 4, 3, 2, 1])  # C₆⁵
                },
                'vector_char': [3, 2, 0, -1, 0, 2],
                'category': 'Cn groups',
                'complex': True,
                'special_notes': 'ε = exp(πi/3) = 1/2 + i√3/2'
            },
            # ---- Cnv groups ----
            # C₂v 群
            'C_2v': {
                'irreps': {
                    'A1': [1, 1, 1, 1],
                    'A2': [1, 1, -1, -1],
                    'B1': [1, -1, 1, -1],
                    'B2': [1, -1, -1, 1]
                },
                'class_sizes': [1, 1, 1, 1],
                'class_names': ['E', 'C₂', 'σv(xz)', 'σv\'(yz)'],
                'order': 4,
                'class_cycles': {
                    0: (1, [0]),  # E
                    1: (2, [0, 1]),  # C₂
                    2: (2, [0, 2]),  # σv(xz)
                    3: (2, [0, 3]),  # σv'(yz)
                    4: (1, [0])   # E (重复)
                },
                'vector_char': [3, -1, 1, 1],
                'category': 'Cnv groups'
            },
            
                    # C₃v 群
            
            # C₃v 群
            'C_3v': {
                'irreps': {
                    'A1': [1, 1, 1],
                    'A2': [1, 1, -1],
                    'E':  [2, -1, 0]
                },
                'class_sizes': [1, 2, 3],
                'class_names': ['E', '2C₃', '3σv'],
                'order': 6,
                'class_cycles': {
                    0: (1, [0]),
                    1: (3, [0, 1, 1]),   # 2C₃: C₃² 同类
                    2: (2, [0, 2])       # 3σv
                },
                'vector_char': [3, 0, 1],
                'category': 'Cnv groups'
            },

            # C₄v 群
            'C_4v': {
                'irreps': {
                    'A1': [1,  1,  1,  1,  1],
                    'A2': [1,  1,  1, -1, -1],
                    'B1': [1, -1,  1,  1, -1],
                    'B2': [1, -1,  1, -1,  1],
                    'E':  [2,  0, -2,  0,  0]
                },
                'class_sizes': [1, 2, 1, 2, 2],
                'class_names': ['E', '2C₄', 'C₂', '2σv', '2σd'],
                'order': 8,
                'class_cycles': {
                    0: (1, [0]),
                    1: (4, [0, 1, 2, 1]), # 2C₄
                    2: (2, [0, 2]),       # C₂
                    3: (2, [0, 3]),       # 2σv
                    4: (2, [0, 4])        # 2σd
                },
                'vector_char': [3, 1, -1, 1, 1],
                'category': 'Cnv groups'
            },

            # C₅v 群
            'C_5v': {
                'irreps': {
                    'A1': [1, 1, 1, 1],
                    'A2': [1, 1, 1, -1],
                    'E1': [2, two_cos72, two_cos144, 0],
                    'E2': [2, two_cos144, two_cos72, 0]
                },
                'class_sizes': [1, 2, 2, 5],
                'class_names': ['E', '2C₅', '2C₅²', '5σv'],
                'order': 10,
                'class_cycles': {
                    0: (1, [0]),
                    1: (5, [0, 1, 2, 2, 1]),   # 2C₅
                    2: (5, [0, 2, 1, 1, 2]),   # 2C₅²
                    3: (2, [0, 3])            # 5σv
                },
                'vector_char': [3, two_cos72 + 1, two_cos144 + 1, 1],
                'category': 'Cnv groups',
                'special_notes': 'η⁺ = (1+√5)/2, η⁻ = (1-√5)/2, 2cos72° = (√5-1)/2, 2cos144° = -(√5+1)/2'
            },

            # C₆v 群
            'C_6v': {
                'irreps': {
                    'A1': [1,  1,  1,  1,  1,  1],
                    'A2': [1,  1,  1,  1, -1, -1],
                    'B1': [1, -1,  1, -1,  1, -1],
                    'B2': [1, -1,  1, -1, -1,  1],
                    'E1': [2,  1, -1, -2,  0,  0],
                    'E2': [2, -1, -1,  2,  0,  0]
                },
                'class_sizes': [1, 2, 2, 1, 3, 3],
                'class_names': ['E', '2C₆', '2C₃', 'C₂', '3σv', '3σd'],
                'order': 12,
                'class_cycles': {
                    0: (1, [0]),
                    1: (6, [0, 1, 2, 3, 2, 1]), # 2C₆
                    2: (3, [0, 2, 2]),         # 2C₃
                    3: (2, [0, 3]),            # C₂
                    4: (2, [0, 4]),            # 3σv
                    5: (2, [0, 5])             # 3σd
                },
                'vector_char': [3, 2, 0, -1, 1, 1],
                'category': 'Cnv groups'
            },
            # ---- Cnh groups ----
            # C₂h 群
            'C_2h': {
                'irreps': {
                    'Ag': [1,  1,  1,  1],
                    'Bg': [1, -1,  1, -1],
                    'Au': [1,  1, -1, -1],
                    'Bu': [1, -1, -1,  1]
                },
                'class_sizes': [1, 1, 1, 1],
                'class_names': ['E', 'C₂', 'i', 'σh'],
                'order': 4,
                'class_cycles': {
                    0: (1, [0]),
                    1: (2, [0, 1]),
                    2: (2, [0, 2]),
                    3: (2, [0, 3])
                },
                'vector_char': [3, -1, -3, 1],
                'category': 'Cnh groups'
            },

            # C₃h 群
            'C_3h': {
                'irreps': {
                    'A\'':   [1, 1, 1, 1, 1, 1],
                    'A\'\'':  [1, 1, 1, -1, -1, -1],
                    'E\'':   [1, omega, omega_squared, 1, omega, omega_squared],
                    'E\'*':  [1, omega_squared, omega, 1, omega_squared, omega],
                    'E\'\'':  [1, omega, omega_squared, -1, -omega, -omega_squared],
                    'E\'\'*': [1, omega_squared, omega, -1, -omega_squared, -omega]
                },
                'class_sizes': [1, 1, 1, 1, 1, 1],
                'class_names': ['E', 'C₃', 'C₃²', 'σh', 'S₃', 'S₃⁵'],
                'order': 6,
                'class_cycles': {
                    0: (1, [0]),
                    1: (3, [0, 1, 2]),
                    2: (3, [0, 2, 1]),
                    3: (2, [0, 3]),
                    4: (6, [0, 4, 2, 3, 1, 5]), # S₃
                    5: (6, [0, 5, 1, 3, 2, 4])  # S₃⁵
                },
                'vector_char': [3, 0, 0, 1, -2, -2],
                'category': 'Cnh groups',
                'complex': True,
                'special_notes': 'ω = exp(2πi/3), ω² = exp(4πi/3)'
            },

            # C₄h 群
            'C_4h': {
                'irreps': {
                    'Ag':  [1,  1,  1,  1,  1,  1,  1,  1],
                    'Bg':  [1, -1,  1, -1,  1, -1,  1, -1],
                    'Eg':  [1,  1j, -1, -1j, 1,  1j, -1, -1j],
                    'Eg*': [1, -1j, -1,  1j, 1, -1j, -1,  1j],
                    'Au':  [1,  1,  1,  1, -1, -1, -1, -1],
                    'Bu':  [1, -1,  1, -1, -1,  1, -1,  1],
                    'Eu':  [1,  1j, -1, -1j, -1, -1j,  1,  1j],
                    'Eu*': [1, -1j, -1,  1j, -1,  1j,  1, -1j]
                },
                'class_sizes': [1, 1, 1, 1, 1, 1, 1, 1],
                'class_names': ['E', 'C₄', 'C₂', 'C₄³', 'i', 'S₄³', 'σh', 'S₄'],
                'order': 8,
                'class_cycles': {
                    0: (1, [0]),
                    1: (4, [0, 1, 2, 3]),
                    2: (2, [0, 2]),
                    3: (4, [0, 3, 2, 1]),
                    4: (2, [0, 4]),
                    5: (4, [0, 5, 2, 7]),
                    6: (2, [0, 6]),
                    7: (4, [0, 7, 2, 5])
                },
                'vector_char': [3, 1, -1, 1, -3, -1, 1, -1],
                'category': 'Cnh groups',
                'complex': True,
                'special_notes': 'i = exp(πi/2)'
            },

            # C₅h 群
            'C_5h': {
                'irreps': {
                    'A\'':   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    "E1''":  [1, epsilon10**1, epsilon10**2, epsilon10**3, epsilon10**4, epsilon10**5, epsilon10**6, epsilon10**7, epsilon10**8, epsilon10**9],
                    "E2'":   [1, epsilon10**2, epsilon10**4, epsilon10**6, epsilon10**8, epsilon10**0, epsilon10**2, epsilon10**4, epsilon10**6, epsilon10**8],
                    "E2''":  [1, epsilon10**3, epsilon10**6, epsilon10**9, epsilon10**2, epsilon10**5, epsilon10**8, epsilon10**1, epsilon10**4, epsilon10**7],
                    "E1'":   [1, epsilon10**4, epsilon10**8, epsilon10**2, epsilon10**6, epsilon10**0, epsilon10**4, epsilon10**8, epsilon10**2, epsilon10**6],
                    "A''":   [1, epsilon10**5, epsilon10**0, epsilon10**5, epsilon10**0, epsilon10**5, epsilon10**0, epsilon10**5, epsilon10**0, epsilon10**5],
                    "E1'*":  [1, epsilon10**6, epsilon10**2, epsilon10**8, epsilon10**4, epsilon10**0, epsilon10**6, epsilon10**2, epsilon10**8, epsilon10**4],
                    "E2''*": [1, epsilon10**7, epsilon10**4, epsilon10**1, epsilon10**8, epsilon10**5, epsilon10**2, epsilon10**9, epsilon10**6, epsilon10**3],
                    "E2'*":  [1, epsilon10**8, epsilon10**6, epsilon10**4, epsilon10**2, epsilon10**0, epsilon10**8, epsilon10**6, epsilon10**4, epsilon10**2],
                    "E1''*": [1, epsilon10**9, epsilon10**8, epsilon10**7, epsilon10**6, epsilon10**5, epsilon10**4, epsilon10**3, epsilon10**2, epsilon10**1],
                },
                'class_sizes': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                'class_names': ['E', 'S₅', 'C₅²', 'S₅³', 'C₅⁴', 'σh', 'C₅', 'S₅⁷', 'C₅³', 'S₅⁹'],
                'order': 10,
                'class_cycles': {
                    0: (1, [0]), # E
                    1: (10, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), # S₅
                    2: (5, [0, 2, 4, 6, 8]), # C₅²
                    3: (10, [0, 3, 6, 9, 2, 5, 8, 1, 4, 7]), # S₅³
                    4: (5, [0, 4, 8, 2, 6]), # C₅⁴
                    5: (2, [0, 5]), # σh
                    6: (5, [0, 6, 2, 8, 4]), # C₅
                    7: (10, [0, 7, 4, 1, 8, 5, 2, 9, 6, 3]), # S₅⁷
                    8: (5, [0, 8, 6, 4, 2]), # C₅³
                    9: (10, [0, 9, 8, 7, 6, 5, 4, 3, 2, 1]), # S₅⁹
                },
                'vector_char': [3, two_cos72 - 1, two_cos144 + 1, two_cos144 - 1, two_cos72 + 1,
                                1, two_cos72 + 1, two_cos144 - 1, two_cos144 + 1, two_cos72 - 1],
                'category': 'Cnh groups',
                'complex': True,
                'special_notes': 'ε = exp(2πi/5), 2cos72° = (√5-1)/2, 2cos144° = -(√5+1)/2'
            },

            # C₆h 群
            'C_6h': {
                'irreps': {
                    'Ag':   [1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
                    'Bg':   [1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1],
                    'E1g':  [1,  epsilon6, -epsilon6_star, -1, -epsilon6,  epsilon6_star, 1,  epsilon6, -epsilon6_star, -1, -epsilon6,  epsilon6_star],
                    'E1g*': [1,  epsilon6_star, -epsilon6, -1, -epsilon6_star,  epsilon6, 1,  epsilon6_star, -epsilon6, -1, -epsilon6_star,  epsilon6],
                    'E2g':  [1, -epsilon6_star, -epsilon6,  1, -epsilon6_star, -epsilon6, 1, -epsilon6_star, -epsilon6,  1, -epsilon6_star, -epsilon6],
                    'E2g*': [1, -epsilon6, -epsilon6_star,  1, -epsilon6, -epsilon6_star, 1, -epsilon6, -epsilon6_star,  1, -epsilon6, -epsilon6_star],
                    'Au':   [1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1],
                    'Bu':   [1, -1,  1, -1,  1, -1, -1,  1, -1,  1, -1,  1],
                    'E1u':  [1,  epsilon6, -epsilon6_star, -1, -epsilon6,  epsilon6_star, -1, -epsilon6,  epsilon6_star, 1,  epsilon6, -epsilon6_star],
                    'E1u*': [1,  epsilon6_star, -epsilon6, -1, -epsilon6_star,  epsilon6, -1, -epsilon6_star,  epsilon6, 1,  epsilon6_star, -epsilon6],
                    'E2u':  [1, -epsilon6_star, -epsilon6,  1, -epsilon6_star, -epsilon6, -1,  epsilon6_star,  epsilon6, -1,  epsilon6_star,  epsilon6],
                    'E2u*': [1, -epsilon6, -epsilon6_star,  1, -epsilon6, -epsilon6_star, -1,  epsilon6,  epsilon6_star, -1,  epsilon6,  epsilon6_star]
                },
                'class_sizes': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                'class_names': ['E', 'C₆', 'C₃', 'C₂', 'C₃²', 'C₆⁵',
                                'i', 'S₃⁵', 'S₆⁵', 'σh', 'S₆', 'S₃'],
                'order': 12,
                'class_cycles': {
                    0:  (1, [0]), # E
                    1:  (6, [0, 1, 2, 3, 4, 5]), # C₆
                    2:  (3, [0, 2, 4]), # C₃
                    3:  (2, [0, 3]), # C₂
                    4:  (3, [0, 4, 2]), # C₃²
                    5:  (6, [0, 5, 4, 3, 2, 1]), # C₆⁵
                    6:  (2, [0, 6]), # i
                    7:  (6, [0, 7, 2, 9, 4, 11]),  # S₃⁵
                    8:  (6, [0, 8, 4, 6, 2, 10]),  # S₆⁵
                    9:  (2, [0, 9]), # σh
                    10: (6, [0, 10, 2, 6, 4, 8]), # S₆
                    11: (6, [0, 11, 4, 9, 2, 7])  # S₃
                },
                'vector_char': [3, 2, 0, -1, 0, 2, -3, -2, 0, 1, 0, -2],
                'category': 'Cnh groups',
                'complex': True,
                'special_notes': 'ε = exp(πi/3)'
            },

            # ---- Dn groups ----
            # D₂ 群
            'D_2': {
                'irreps': {
                    'A':  [1,  1,  1,  1],
                    'B1': [1,  1, -1, -1],
                    'B2': [1, -1, -1,  1],
                    'B3': [1, -1,  1, -1]
                },
                'class_sizes': [1, 1, 1, 1],
                'class_names': ['E', 'C₂(z)', 'C₂(y)', 'C₂(x)'],
                'order': 4,
                'class_cycles': {
                    0: (1, [0]),
                    1: (2, [0, 1]),
                    2: (2, [0, 2]),
                    3: (2, [0, 3])
                },
                'vector_char': [3, -1, -1, -1],
                'category': 'Dn groups'
            },
            
            # D_3 群
            'D_3': {
                'irreps': {
                    'A1': [1, 1, 1],
                    'A2': [1, 1, -1],
                    'E': [2, -1, 0]
                },
                'class_sizes': [1, 2, 3],
                'class_names': ['E', '2C₃', '3C₂'],
                'order': 6,
                'class_cycles': {
                    0: (1, [0]),  # E
                    1: (3, [0, 1, 1]),  # 2C₃
                    2: (2, [0, 2])  # 3C₂
                },
                'vector_char': [3, 0, -1],
                'category': 'Dn groups'
            },
            
            # D₄ 群
            'D_4': {
                'irreps': {
                    'A1': [1, 1, 1, 1, 1],
                    'A2': [1, 1, 1, -1, -1],
                    'B1': [1, -1, 1, 1, -1],
                    'B2': [1, -1, 1, -1, 1],
                    'E':  [2, 0, -2, 0, 0]
                },
                'class_sizes': [1, 2, 1, 2, 2],
                'class_names': ['E', '2C₄', 'C₂', '2C₂\'', '2C₂\'\''],
                'order': 8,
                'class_cycles': {
                    0: (1, [0]),           # E
                    1: (4, [0, 1, 2, 1]), # 2C₄
                    2: (2, [0, 2]),       # C₂
                    3: (2, [0, 3]),       # 2C₂'
                    4: (2, [0, 4])        # 2C₂''
                },
                'vector_char': [3, 1, -1, -1, -1],
                'category': 'Dn groups'
            },

            # D₅ 群
            'D_5': {
                'irreps': {
                    'A1': [1, 1, 1, 1],
                    'A2': [1, 1, 1, -1],
                    'E1': [2, two_cos72, two_cos144, 0],
                    'E2': [2, two_cos144, two_cos72, 0]
                },
                'class_sizes': [1, 2, 2, 5],
                'class_names': ['E', '2C₅', '2C₅²', '5C₂'],
                'order': 10,
                'class_cycles': {
                    0: (1, [0]),           # E
                    1: (5, [0, 1, 2, 2, 1]), # 2C₅
                    2: (5, [0, 2, 1, 1, 2]), # 2C₅²
                    3: (2, [0, 3])         # 5C₂
                },
                'vector_char': [3, eta_plus, eta_minus, -1],
                'category': 'Dn groups',
                'special_notes': 'η⁺ = (1+√5)/2 ≈ 1.618, η⁻ = (1-√5)/2 ≈ -0.618'
            },

            # D₆ 群
            'D_6': {
                'irreps': {
                    'A1': [1, 1, 1, 1, 1, 1],
                    'A2': [1, 1, 1, 1, -1, -1],
                    'B1': [1, -1, 1, -1, 1, -1],
                    'B2': [1, -1, 1, -1, -1, 1],
                    'E1': [2, 1, -1, -2, 0, 0],
                    'E2': [2, -1, -1, 2, 0, 0]
                },
                'class_sizes': [1, 2, 2, 1, 3, 3],
                'class_names': ['E', '2C₆', '2C₃', 'C₂', '3C₂\'', '3C₂\'\''],
                'order': 12,
                'class_cycles': {
                    0: (1, [0]),           # E
                    1: (6, [0, 1, 2, 3, 2, 1]), # 2C₆
                    2: (3, [0, 2, 2]),     # 2C₃
                    3: (2, [0, 3]),        # C₂
                    4: (2, [0, 4]),        # 3C₂'
                    5: (2, [0, 5])         # 3C₂''
                },
                'vector_char': [3, 2, 0, -1, -1, -1],
                'category': 'Dn groups'
            },
            # ---- Dnh groups ----
            # D₂h 群
            'D_2h': {
                'irreps': {
                    'Ag':  [1,  1,  1,  1,  1,  1,  1,  1],
                    'B1g': [1,  1, -1, -1,  1,  1, -1, -1],
                    'B2g': [1, -1, -1,  1,  1, -1,  1, -1],
                    'B3g': [1, -1,  1, -1,  1, -1, -1,  1],
                    'Au':  [1,  1,  1,  1, -1, -1, -1, -1],
                    'B1u': [1,  1, -1, -1, -1, -1,  1,  1],
                    'B2u': [1, -1, -1,  1, -1,  1, -1,  1],
                    'B3u': [1, -1,  1, -1, -1,  1,  1, -1]
                },
                'class_sizes': [1, 1, 1, 1, 1, 1, 1, 1],
                'class_names': ['E', 'C₂', 'C₂\'', 'C₂\'\'', 'i', 'σh', 'σv', 'σd'],
                'order': 8,
                'class_cycles': {
                    0: (1, [0]),
                    1: (2, [0, 1]),
                    2: (2, [0, 2]),
                    3: (2, [0, 3]),
                    4: (2, [0, 4]),
                    5: (2, [0, 5]),
                    6: (2, [0, 6]),
                    7: (2, [0, 7])
                },
                'vector_char': [3, -1, -1, -1, -3, 1, 1, 1],
                'category': 'Dnh groups'
            },
            
            # D_3h 群
            'D_3h': {
                'irreps': {
                    'A1\'': [1, 1, 1, 1, 1, 1],
                    'A2\'': [1, 1, -1, 1, 1, -1],
                    'E\'': [2, -1, 0, 2, -1, 0],
                    'A1\'\'': [1, 1, 1, -1, -1, -1],
                    'A2\'\'': [1, 1, -1, -1, -1, 1],
                    'E\'\'': [2, -1, 0, -2, 1, 0]
                },
                'class_sizes': [1, 2, 3, 1, 2, 3],
                'class_names': ['E', '2C₃', '3C₂\'','σh', '2S₃', '3σv'],
                'order': 12,
                'class_cycles': {
                    0: (1, [0]),  # E
                    1: (3, [0, 1, 1]),  # 2C₃
                    2: (2, [0, 2]),  # 3C₂
                    3: (2, [0, 3]), # σh
                    4: (6, [0, 4, 1, 3, 1, 4]), # 2S₃
                    5: (2, [0, 5]) # 3σv
                },
                'vector_char': [3, 0, -1, 1, -2, 1],
                'category': 'Dnh groups'
            },

            # D₄h 群
            'D_4h': {
                'irreps': {
                    'A1g': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    'A2g': [1, 1, 1, -1, -1, 1, 1, 1, -1, -1],
                    'B1g': [1, -1, 1, 1, -1, 1, -1, 1, 1, -1],
                    'B2g': [1, -1, 1, -1, 1, 1, -1, 1, -1, 1],
                    'Eg': [2, 0, -2, 0, 0, 2, 0, -2, 0, 0],
                    'A1u': [1, 1, 1, 1, 1, -1, -1, -1, -1, -1],
                    'A2u': [1, 1, 1, -1, -1, -1, -1, -1, 1, 1],
                    'B1u': [1, -1, 1, 1, -1, -1, 1, -1, -1, 1],
                    'B2u': [1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
                    'Eu': [2, 0, -2, 0, 0, -2, 0, 2, 0, 0]
                },
                'class_sizes': [1, 2, 1, 2, 2, 1, 2, 1, 2, 2],
                'class_names': ['E', '2C₄', 'C₂', '2C₂\'', '2C₂\'\'', 'i', '2S₄', 'σh', '2σv', '2σd'],
                'order': 16,
                'class_cycles': {
                    0: (1, [0]),  # E
                    1: (4, [0, 1, 2, 1]),  # 2C₄: 循环为[E, C₄, C₂, C₄]
                    2: (2, [0, 2]),  # C₂
                    3: (2, [0, 3]),  # 2C₂'
                    4: (2, [0, 4]),  # 2C₂''
                    5: (2, [0, 5]),  # i
                    6: (4, [0, 6, 2, 6]),  # 2S₄: 循环为[E, S₄, C₂, S₄]
                    7: (2, [0, 7]),  # σh
                    8: (2, [0, 8]),  # 2σv
                    9: (2, [0, 9])   # 2σd
                },
                'vector_char': [3, 1, -1, -1, -1, -3, -1, 1, 1, 1],
                'category': 'Dnh groups'
            },
            
            # D₅h 群
            'D_5h': {
                'irreps': {
                    'A1\'':  [1, 1, 1, 1, 1, 1, 1, 1],
                    'A2\'':  [1, 1, 1, -1, 1, 1, 1, -1],
                    'E1\'':  [2, two_cos72, two_cos144, 0, 2, two_cos72, two_cos144, 0],
                    'E2\'':  [2, two_cos144, two_cos72, 0, 2, two_cos144, two_cos72, 0],
                    'A1\'\'': [1, 1, 1, 1, -1, -1, -1, -1],
                    'A2\'\'': [1, 1, 1, -1, -1, -1, -1, 1],
                    'E1\'\'': [2, two_cos72, two_cos144, 0, -2, -two_cos72, -two_cos144, 0],
                    'E2\'\'': [2, two_cos144, two_cos72, 0, -2, -two_cos144, -two_cos72, 0]
                },
                'class_sizes': [1, 2, 2, 5, 1, 2, 2, 5],
                'class_names': ['E', '2C₅', '2C₅²', '5C₂', 'σh', '2S₅', '2S₅³', '5σv'],
                'order': 20,
                'class_cycles': {
                    0: (1, [0]),                     # E
                    1: (5, [0, 1, 2, 2, 1]),         # 2C₅
                    2: (5, [0, 2, 1, 1, 2]),         # 2C₅²
                    3: (2, [0, 3]),                 # 5C₂
                    4: (2, [0, 4]),                 # σh
                    5: (10, [0, 5, 2, 6, 1, 4, 1, 6, 2, 5]), # 2S₅
                    6: (10, [0, 6, 1, 5, 2, 4, 2, 5, 1, 6]), # 2S₅³
                    7: (2, [0, 7])                  # 5σv
                },
                'vector_char': [3, eta_plus, eta_minus, -1, 1, two_cos72-1, two_cos144-1, 1],
                'category': 'Dnh groups',
                'special_notes': 'η⁺ = (1+√5)/2 ≈1.618, η⁻ = (1-√5)/2 ≈-0.618, 2cos72°≈0.618, 2cos144°≈-1.618'
            },
            
            # D₆h 群
            'D_6h': {
                'irreps': {
                    'A1g': [1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
                    'A2g': [1,  1,  1,  1, -1, -1,  1,  1,  1,  1, -1, -1],
                    'B1g': [1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1],
                    'B2g': [1, -1,  1, -1, -1,  1,  1, -1,  1, -1, -1,  1],
                    'E1g': [2,  1, -1, -2,  0,  0,  2,  1, -1, -2,  0,  0],
                    'E2g': [2, -1, -1,  2,  0,  0,  2, -1, -1,  2,  0,  0],
                    'A1u': [1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1],
                    'A2u': [1,  1,  1,  1, -1, -1, -1, -1, -1, -1,  1,  1],
                    'B1u': [1, -1,  1, -1,  1, -1, -1,  1, -1,  1, -1,  1],
                    'B2u': [1, -1,  1, -1, -1,  1, -1,  1, -1,  1,  1, -1],
                    'E1u': [2,  1, -1, -2,  0,  0, -2, -1,  1,  2,  0,  0],
                    'E2u': [2, -1, -1,  2,  0,  0, -2,  1,  1, -2,  0,  0]
                },
                'class_sizes': [1, 2, 2, 1, 3, 3, 1, 2, 2, 1, 3, 3],
                'class_names': ['E', '2C₆', '2C₃', 'C₂', '3C₂\'', '3C₂\'\'',
                                'i', '2S₃', '2S₆', 'σh', '3σd', '3σv'],
                'order': 24,
                'class_cycles': {
                    0:  (1, [0]),                  # E
                    1:  (6, [0, 1, 2, 3, 2, 1]),   # 2C₆
                    2:  (3, [0, 2, 2]),           # 2C₃
                    3:  (2, [0, 3]),              # C₂
                    4:  (2, [0, 4]),              # 3C₂'
                    5:  (2, [0, 5]),              # 3C₂''
                    6:  (2, [0, 6]),              # i
                    7:  (6, [0, 7, 2, 9, 2, 7]),  # 2S₃
                    8:  (6, [0, 8, 2, 6, 2, 8]),  # 2S₆
                    9:  (2, [0, 9]),              # σh
                    10: (2, [0, 10]),             # 3σd
                    11: (2, [0, 11])              # 3σv
                },
                'vector_char': [3, 2, 0, -1, -1, -1, -3, -2, 0, 1, 1, 1],
                'category': 'Dnh groups'
            },

            # ---- Dnd groups ----
            # D₂d 群
            'D_2d': {
                'irreps': {
                    'A1': [1,  1,  1,  1,  1],
                    'A2': [1,  1,  1, -1, -1],
                    'B1': [1, -1,  1,  1, -1],
                    'B2': [1, -1,  1, -1,  1],
                    'E':  [2,  0, -2,  0,  0]
                },
                'class_sizes': [1, 2, 1, 2, 2],
                'class_names': ['E', '2S₄', 'C₂', '2C₂\'', '2σd'],
                'order': 8,
                'class_cycles': {
                    0: (1, [0]),
                    1: (4, [0, 1, 2, 1]),   # S₄: 阶4，循环 [E, S₄, C₂, S₄]
                    2: (2, [0, 2]),
                    3: (2, [0, 3]),         # C₂'
                    4: (2, [0, 4])          # σd
                },
                'vector_char': [3, -1, -1, -1, 1],
                'category': 'Dnd groups'
            },
            
            # D_3d 群
            'D_3d': {
                'irreps': {
                    'A1g': [1, 1, 1, 1, 1, 1],
                    'A2g': [1, 1, -1, 1, 1, -1],
                    'Eg': [2, -1, 0, 2, -1, 0],
                    'A1u': [1, 1, 1, -1, -1, -1],
                    'A2u': [1, 1, -1, -1, -1, 1],
                    'Eu': [2, -1, 0, -2, 1, 0]
                },
                'class_sizes': [1, 2, 3, 1, 2, 3],
                'class_names': ['E', '2C₃', '3C₂', 'i', '2S₆', '3σd'],
                'order': 12,
                'class_cycles': {
                    0: (1, [0]),  # E
                    1: (3, [0, 1, 1]),  # 2C₃
                    2: (2, [0, 2]),  # 3C₂
                    3: (2, [0, 3]),  # i
                    4: (6, [0, 4, 1, 3, 1, 4]),  # 2S₆: 阶为6，循环为[E, S₆, C₃, i, C₃², S₆]
                    5: (2, [0, 5])  # 3σd
                },
                'vector_char': [3, 0, -1, -3, 0, 1],
                'category': 'Dnd groups'
            },
            # ---- Dnd groups ----
            # D₄d 群
            'D_4d': {
                'irreps': {
                    'A1': [1,  1,  1,  1,  1,  1,  1],
                    'A2': [1,  1,  1,  1,  1, -1, -1],
                    'B1': [1, -1,  1, -1,  1,  1, -1],
                    'B2': [1, -1,  1, -1,  1, -1,  1],
                    'E1': [2,  sqrt2, 0, -sqrt2, -2, 0, 0],
                    'E2': [2,  0, -2,  0,  2, 0, 0],
                    'E3': [2, -sqrt2, 0,  sqrt2, -2, 0, 0]
                },
                'class_sizes': [1, 2, 2, 2, 1, 4, 4],
                'class_names': ['E', '2S₈', '2C₄', '2S₈³', 'C₂', '4C₂\'', '4σd'],
                'order': 16,
                'class_cycles': {
                    # 类索引：0:E, 1:2S₈, 2:2C₄, 3:2S₈³, 4:C₂, 5:4C₂', 6:4σd
                    0: (1, [0]),                    # E 阶1
                    1: (8, [0, 1, 2, 3, 4, 3, 2, 1]), # 2S₈ 阶8，幂次映射
                    2: (4, [0, 2, 4, 2]),            # 2C₄ 阶4
                    3: (8, [0, 3, 2, 1, 4, 1, 2, 3]), # 2S₈³ 阶8
                    4: (2, [0, 4]),                  # C₂ 阶2
                    5: (2, [0, 5]),                  # 4C₂' 阶2
                    6: (2, [0, 6])                   # 4σd 阶2
                },
                # 向量表示特征标（三维空间向量表示，分解为 B₂ + E₁）
                'vector_char': [3, -1 + sqrt2, 1, -1 - sqrt2, -1, -1, 1],
                'category': 'Dnd groups',
                'special_notes': '√2 = 1.41421356'
            },

            # D₅d 群
            'D_5d': {
                'irreps': {
                    'A1g': [1, 1, 1, 1, 1, 1, 1, 1],
                    'A2g': [1, 1, 1, -1, 1, 1, 1, -1],
                    'E1g': [2, two_cos72, two_cos144, 0, 2, two_cos144, two_cos72, 0],
                    'E2g': [2, two_cos144, two_cos72, 0, 2, two_cos72, two_cos144, 0],
                    'A1u': [1, 1, 1, 1, -1, -1, -1, -1],
                    'A2u': [1, 1, 1, -1, -1, -1, -1, 1],
                    'E1u': [2, two_cos72, two_cos144, 0, -2, -two_cos144, -two_cos72, 0],
                    'E2u': [2, two_cos144, two_cos72, 0, -2, -two_cos72, -two_cos144, 0]
                },
                'class_sizes': [1, 2, 2, 5, 1, 2, 2, 5],
                'class_names': ['E', '2C₅', '2C₅²', '5C₂', 'i',  '2S₁₀³','2S₁₀', '5σd'],
                'order': 20,
                'class_cycles': {
                    0: (1, [0]),                     # E
                    1: (5, [0, 1, 2, 2, 1]),         # 2C₅
                    2: (5, [0, 2, 1, 1, 2]),         # 2C₅²
                    3: (2, [0, 3]),                 # 5C₂
                    4: (2, [0, 4]),                 # i
                    5: (10, [0, 5, 1, 6, 2, 4, 2, 6, 1, 5]), # 2S₁₀
                    6: (10, [0, 6, 2, 5, 1, 4, 1, 5, 2, 6]), # 2S₁₀³
                    7: (2, [0, 7])                  # 5σd
                },
                'vector_char': [3, eta_plus, eta_minus, -1, -3, -eta_minus, -eta_plus, 1],
                'category': 'Dnd groups',
                'special_notes': 'η⁺ = (1+√5)/2, η⁻ = (1-√5)/2, 2cos72°≈0.618, 2cos144°≈-1.618'
            },

            # D₆d 群
            'D_6d': {
                'irreps': {
                    'A1': [1,  1,  1,  1,  1,  1,  1,  1,  1,],
                    'A2': [1,  1,  1,  1,  1,  1,  1, -1, -1],
                    'B1': [1, -1,  1, -1,  1, -1,  1,  1, -1],
                    'B2': [1, -1,  1, -1,  1, -1,  1, -1,  1],
                    'E1': [2,  sqrt3,  1,  0, -1, -sqrt3, -2,  0,  0],
                    'E2': [2,  1, -1, -2, -1,  1,  2,  0,  0],
                    'E3': [2,  0, -2,  0,  2,  0,  -2,  0,  0],
                    'E4': [2, -1, -1,  2, -1, -1,  2,  0,  0],
                    'E5': [2, -sqrt3,  1,  0, -1,  sqrt3, -2,  0,  0]
                },
                'class_sizes': [1, 2, 2, 2, 2, 2, 1, 6, 6],
                'class_names': ['E', '2S₁₂', '2C₆', '2S₄', '2C₃', '2S₁₂⁵',
                                'C₂', '6C₂\'', '6σd\''],
                'order': 24,
                'class_cycles': {
                    0:  (1, [0]),                  # E
                    1:  (12, [0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1]), # 2S₁₂
                    2:  (6,  [0, 2, 4, 6, 4, 2]), # 2C₆
                    3:  (4,  [0, 3, 6, 3]),       # 2S₄
                    4:  (3,  [0, 4, 4]),          # 2C₃
                    5:  (12, [0, 5, 2, 3, 4, 1, 6, 1, 4, 3, 2, 5]), # 2S₁₂⁵
                    6:  (2,  [0, 6]),             # C₂
                    7:  (2,  [0, 7]),             # 6C₂'
                    8:  (2,  [0, 8]),             # 6σd'
                },
                'vector_char': [3, -1 + sqrt3, 2, -1, 0, -1 - sqrt3, -1, -1,  1],
                'category': 'Dnd groups',
                'special_notes': '√3 = 1.73205081'
            },

            # ---- Sn groups ----
            # Cᵢ 群（又名 S₂）
            'C_i': {
                'irreps': {
                    'Ag': [1,  1],
                    'Au': [1, -1]
                },
                'class_sizes': [1, 1],
                'class_names': ['E', 'i'],
                'order': 2,
                'class_cycles': {
                    0: (1, [0]),
                    1: (2, [0, 1])
                },
                'vector_char': [3, -3],
                'category': 'Sn groups'
            },
            
                    # ---- Sn groups ----
            # S₄ 群
            'S_4': {
                'irreps': {
                    'A':  [1,  1,  1,  1],
                    'B':  [1, -1,  1, -1],
                    'E':  [1,  1j, -1, -1j],
                    'E*': [1, -1j, -1,  1j]
                },
                'class_sizes': [1, 1, 1, 1],
                'class_names': ['E', 'S₄', 'C₂', 'S₄³'],
                'order': 4,
                'class_cycles': {
                    0: (1, [0]),           # E
                    1: (4, [0, 1, 2, 3]), # S₄
                    2: (2, [0, 2]),       # C₂
                    3: (4, [0, 3, 2, 1])  # S₄³
                },
                'vector_char': [3, -1, -1, -1],
                'category': 'Sn groups',
                'complex': True,
                'special_notes': 'i = exp(πi/2)'
            },

            # S₆ 群
            'S_6': {
                'irreps': {
                    'Ag':  [1,  1,  1,  1,  1,  1],
                    'Eg':  [1,  omega, omega_squared, 1,  omega, omega_squared],
                    'Eg*': [1,  omega_squared, omega, 1,  omega_squared, omega],
                    'Au':  [1,  1,  1, -1, -1, -1],
                    'Eu':  [1,  omega, omega_squared, -1, -omega, -omega_squared],
                    'Eu*': [1,  omega_squared, omega, -1, -omega_squared, -omega]
                },
                'class_sizes': [1, 1, 1, 1, 1, 1],
                'class_names': ['E', 'C₃', 'C₃²', 'i', 'S₆⁵', 'S₆'],
                'order': 6,
                'class_cycles': {
                    0: (1, [0]),           # E
                    1: (3, [0, 1, 2]),     # C₃
                    2: (3, [0, 2, 1]),     # C₃²
                    3: (2, [0, 3]),        # i
                    4: (6, [0, 4, 2, 3, 1, 5]), # S₆⁵
                    5: (6, [0, 5, 1, 3, 2, 4])  # S₆
                },
                'vector_char': [3, 0, 0, -3, 0, 0],
                'category': 'Sn groups',
                'complex': True,
                'special_notes': 'ω = exp(2πi/3), ω² = exp(4πi/3)'
            },
            # ---- Cubic groups ----
            # T 群
            'T': {
                'irreps': {
                    'A': [1, 1, 1, 1],
                    'E': [1, omega, omega_squared, 1],
                    'E\'': [1, omega_squared, omega, 1],
                    'T': [3, 0, 0, -1]
                },
                'class_sizes': [1, 4, 4, 3],
                'class_names': ['E', '4C₃', '4C₃²', '3C₂'],
                'order': 12,
                'special_notes': 'ω = exp(2πi/3), ω² = exp(4πi/3)',
                'complex': True,
                'class_cycles': {
                    0: (1, [0]),  # E
                    1: (3, [0, 1, 2]),  # 4C₃: 循环为[E, C₃, C₃²]
                    2: (3, [0, 2, 1]),   # 4C₃²: 循环为[E, C₃², C₃]
                    3: (2, [0, 3])
                },
                'vector_char': [3, 0, 0, -1],
                'category': 'Cubic groups'
            },
                   
            # T_d 群
            'T_d': {
                'irreps': {
                    'A1': [1, 1, 1, 1, 1],
                    'A2': [1, 1, 1, -1, -1],
                    'E': [2, -1, 2, 0, 0],
                    'T1': [3, 0, -1, 1, -1],
                    'T2': [3, 0, -1, -1, 1]
                },
                'class_sizes': [1, 8, 3, 6, 6],
                'class_names': ['E', '8C₃', '3C₂', '6S₄', '6σd'],
                'order': 24,
                'class_cycles': {
                    0: (1, [0]),  # E
                    1: (3, [0, 1, 1]),  # 8C₃
                    2: (2, [0, 2]),  # 3C₂
                    3: (4, [0, 3, 2, 3]),  # 6S₄
                    4: (2, [0, 4])  # 6σd
                },
                'vector_char': [3, 0, -1, -1, 1],
                'category': 'Cubic groups'
            },
            
            # T_h 群
            'T_h': {
                'irreps': {
                    # 偶表示 (g)
                    'Ag':  [1, 1, 1, 1, 1, 1, 1, 1],
                    'Eg': [1, omega, omega_squared, 1, 1, omega, omega_squared, 1],
                    "E'g":[1, omega_squared, omega, 1, 1, omega_squared, omega, 1],
                    'Tg':  [3, 0, 0, -1, 3, 0, 0, -1],
                    # 奇表示 (u)
                    'Au':  [1, 1, 1, 1, -1, -1, -1, -1],
                    'Eu': [1, omega, omega_squared, 1, -1, -omega, -omega_squared, -1],
                    "E'u":[1, omega_squared, omega, 1, -1, -omega_squared, -omega, -1],
                    'Tu':  [3, 0, 0, -1, -3, 0, 0, 1],
                },
                'class_sizes': [1, 4, 4, 3, 1, 4, 4, 3],
                'class_names': ['E', '4C₃', '4C₃²', '3C₂', 'i', '4S₆', '4S₆⁵', '3σₕ'],
                'order': 24,
                'special_notes': 'ω = exp(2πi/3), ω² = exp(4πi/3)',
                'complex': True,
                'class_cycles': {
                    0: (1, [0]),                # E
                    1: (3, [0, 1, 2]),          # 4C₃
                    2: (3, [0, 2, 1]),          # 4C₃²
                    3: (2, [0, 3]),             # 3C₂
                    4: (2, [0, 4]),             # i
                    5: (6, [0, 5, 2, 4, 1, 6]), # 4S₆  (S₆ = i·C₃)
                    6: (6, [0, 6, 1, 4, 2, 5]), # 4S₆⁵ (S₆⁵ = i·C₃²)
                    7: (2, [0, 7]),             # 3σₕ (σₕ = i·C₂)
                },
                'vector_char': [3, 0, 0, -1, -3, 0, 0, 1],   # 对应 Tu 表示
                'category': 'Cubic groups'
            },

            # O 群
            'O': {
                'irreps': {
                    'A1': [1, 1, 1, 1, 1],
                    'A2': [1, 1, 1, -1, -1],
                    'E': [2, -1, 2, 0, 0],
                    'T1': [3, 0, -1, -1, 1],
                    'T2': [3, 0, -1, 1, -1]
                },
                'class_sizes': [1, 8, 3, 6, 6],
                'class_names': ['E', '8C₃', '3C₂', '6C₂\'', '6C₄'],
                'order': 24,
                'class_cycles': {
                    0: (1, [0]),  # E
                    1: (3, [0, 1, 1]),  # 8C₃
                    2: (2, [0, 2]),  # 3C₂
                    3: (2, [0, 3]),  # 6C₂'
                    4: (4, [0, 4, 2, 4])  # 6C₄: 阶为4，循环为[E, C₄, C₂, C₄]
                },
                'vector_char': [3, 0, -1, -1, 1],   # 纯旋转，χ = 1+2cosθ
                'category': 'Cubic groups'
            },
            
            # O_h 群
            'O_h': {
                'irreps': {
                    'A1g': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    'A2g': [1, 1, -1, -1, 1, 1, -1, 1, 1, -1],
                    'Eg': [2, -1, 0, 0, 2, 2, 0, -1, 2, 0],
                    'T1g': [3, 0, -1, 1, -1, 3, 1, 0, -1, -1],
                    'T2g': [3, 0, 1, -1, -1, 3, -1, 0, -1, 1],
                    'A1u': [1, 1, 1, 1, 1, -1, -1, -1, -1, -1],
                    'A2u': [1, 1, -1, -1, 1, -1, 1, -1, -1, 1],
                    'Eu': [2, -1, 0, 0, 2, -2, 0, 1, -2, 0],
                    'T1u': [3, 0, -1, 1, -1, -3, -1, 0, 1, 1],
                    'T2u': [3, 0, 1, -1, -1, -3, 1, 0, 1, -1]
                },
                'class_sizes': [1, 8, 6, 6, 3, 1, 6, 8, 3, 6],
                'class_names': ['E', '8C₃', '6C₂', '6C₄', '3C₂\'', 'i', '6S₄', '8S₆', '3σh', '6σd'],
                'order': 48,
                # 定义每个共轭类的循环子群信息
                'class_cycles': {
                    0: (1, [0]),  # E: 阶为1，循环为[E]
                    1: (3, [0, 1, 1]),  # 8C₃: 阶为3，循环为[E, C₃, C₃²]，其中C₃和C₃²在同一共轭类
                    2: (2, [0, 2]),  # 6C₂: 阶为2，循环为[E, C₂]
                    3: (4, [0, 3, 4, 3]),  # 6C₄: 阶为4，循环为[E, C₄, C₂', C₄]
                    4: (2, [0, 4]),  # 3C₂': 阶为2，循环为[E, C₂']
                    5: (2, [0, 5]),  # i: 阶为2，循环为[E, i]
                    6: (4, [0, 6, 4, 6]),  # 6S₄: 阶为4，循环为[E, S₄, C₂, S₄]
                    7: (6, [0, 7, 1, 5, 1, 7]),  # 8S₆: 阶为6，循环为[E, S₆, C₃, i, C₃², S₆]
                    8: (2, [0, 8]),  # 3σh: 阶为2，循环为[E, σh]
                    9: (2, [0, 9])   # 6σd: 阶为2，循环为[E, σd]
                },
                # 向量表示（三维旋转/反射表示）的特征标
                'vector_char': [3, 0, -1, 1, -1, -3, -1, 0, 1, 1],
                'category': 'Cubic groups'
            }
        }
        
        for name, data in tables_data.items():
            self.tables[name] = CharacterTable.from_dict(name, data)
    
    def get_table(self, name: str) -> CharacterTable:
        """Retrieve a character table"""
        if name not in self.tables:
            raise ValueError(f"Unknown group: {name}")
        return self.tables[name]
    
    def list_groups(self) -> List[str]:
        """List all available groups"""
        return list(self.tables.keys())
    
    def get_groups_by_category(self, category: str) -> List[str]:
        """Get groups in a specific category"""
        return [name for name, table in self.tables.items() 
                if table.category == category]

    def get_sorted_groups(self) -> List[str]:
        """Get groups sorted by category and order"""
        groups = list(self.tables.keys())
        return sorted(groups, key=lambda name: (
            CATEGORY_ORDER.index(self.tables[name].category) 
            if self.tables[name].category in CATEGORY_ORDER else len(CATEGORY_ORDER),
            self.tables[name].order
        ))