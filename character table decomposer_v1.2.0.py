"""
特征标计算器 (Character Table Decomposer)
==========================================
一个用于点群特征标计算的交互式Python程序，支持多种点群，
可计算张量积、对称积、反对称积、球谐函数表示等，
并可将任意可约表示分解为不可约表示的直和。

版本: 1.2.0
作者: Jianwen Ma
日期: 2026-02-13
版权: Copyright (c) 2026 Kulinkovich-1234
许可证: MIT License (详见下文)
"""
__author__ = "Jianwen Ma"
__version__ = "1.2.0"
__license__ = "MIT"
__copyright__ = "Copyright (c) 2026 Kulinkovich-1234"

import numpy as np
import json
from itertools import combinations, product
from collections import Counter
import math

class CharacterTableDecomposer:
    def __init__(self):
        # 存储所有群的特征标表
        self.character_tables = {}
        
        # 存储用户定义的特征标（按群分组）
        self.stored_characters = {}
        
        # 初始化所有群的特征标表
        self._init_all_tables()
        
        # 默认使用O_h群
        self.current_group = 'O_h'
        self.set_group('O_h')
        
        # 加载保存的特征标
        self.load_characters()
        
    def _init_all_tables(self):
        """初始化所有群的特征标表，并为每个群添加向量表示的特征标 vector_char"""
        import cmath
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

        # ---- Nonaxial groups ----
        # C₁ 群
        self.character_tables['C_1'] = {
            'irreps': {'A': [1]},
            'class_sizes': [1],
            'class_names': ['E'],
            'order': 1,
            'class_cycles': {0: (1, [0])},
            'vector_char': [3],
            'category': 'Nonaxial groups'
        }

        # Cₛ 群（又名 C₁h）
        self.character_tables['C_s'] = {
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
        }

        # ---- Cn groups ----
        # C₂ 群
        self.character_tables['C_2'] = {
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
        }

        # C₃ 群
        self.character_tables['C_3'] = {
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
        }

        # C₄ 群
        self.character_tables['C_4'] = {
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
        }

        # C₅ 群
        self.character_tables['C_5'] = {
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
        }

        # C₆ 群
        self.character_tables['C_6'] = {
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
        }
        # ---- Cnv groups ----
        # C₂v 群
        self.character_tables['C_2v'] = {
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
        }
        
                # C₃v 群
        
        # C₃v 群
        self.character_tables['C_3v'] = {
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
        }

        # C₄v 群
        self.character_tables['C_4v'] = {
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
        }

        # C₅v 群
        self.character_tables['C_5v'] = {
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
        }

        # C₆v 群
        self.character_tables['C_6v'] = {
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
        }
        # ---- Cnh groups ----
        # C₂h 群
        self.character_tables['C_2h'] = {
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
        }

        # C₃h 群
        self.character_tables['C_3h'] = {
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
        }

        # C₄h 群
        self.character_tables['C_4h'] = {
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
        }

        # C₅h 群
        self.character_tables['C_5h'] = {
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
        }

        # C₆h 群
        self.character_tables['C_6h'] = {
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
        }

        # ---- Dn groups ----
        # D₂ 群
        self.character_tables['D_2'] = {
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
        }
        
        # D_3 群
        self.character_tables['D_3'] = {
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
        }
        
        # D₄ 群
        self.character_tables['D_4'] = {
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
        }

        # D₅ 群
        self.character_tables['D_5'] = {
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
        }

        # D₆ 群
        self.character_tables['D_6'] = {
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
        }
        # ---- Dnh groups ----
        # D₂h 群
        self.character_tables['D_2h'] = {
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
        }
        
        # D_3h 群
        self.character_tables['D_3h'] = {
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
        }

        # D₄h 群
        self.character_tables['D_4h'] = {
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
        }
        
        # D₅h 群
        self.character_tables['D_5h'] = {
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
        }
        
        # D₆h 群
        self.character_tables['D_6h'] = {
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
        }

        # ---- Dnd groups ----
        # D₂d 群
        self.character_tables['D_2d'] = {
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
        }
        
        # D_3d 群
        self.character_tables['D_3d'] = {
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
        }
        # ---- Dnd groups ----
        # D₄d 群
        self.character_tables['D_4d'] = {
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
        }

        # D₅d 群
        self.character_tables['D_5d'] = {
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
        }

        # D₆d 群
        self.character_tables['D_6d'] = {
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
        }

        # ---- Sn groups ----
        # Cᵢ 群（又名 S₂）
        self.character_tables['C_i'] = {
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
        }
        
                # ---- Sn groups ----
        # S₄ 群
        self.character_tables['S_4'] = {
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
        }

        # S₆ 群
        self.character_tables['S_6'] = {
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
        }
        # ---- Cubic groups ----
        # T 群
        self.character_tables['T'] = {
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
        }
               
        # T_d 群
        self.character_tables['T_d'] = {
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
        }
        
        # T_h 群
        self.character_tables['T_h'] = {
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
        }

        # O 群
        self.character_tables['O'] = {
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
        }
        
        # O_h 群
        self.character_tables['O_h'] = {
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

    # ============== 新增：统一来源信息处理方法 ==============
    def get_source_display(self, source, name):
        """
        根据来源类型和名称生成统一的显示字符串。
        用于在界面中友好显示特征标的来源。
        """
        if source == 'stored':
            return name
        elif source == 'irrep':
            return f"不可约表示 {name}"
        elif source == 'spherical_harmonics':
            return f"球谐函数 {name}"
        elif source == 'polynomial':
            return f"多项式 {name}"
        elif source == 'manual':
            return "手动输入"
        else:
            return name  # 降级处理

    def get_default_name(self, source_type, source_name):
        """
        根据来源类型和名称生成存储时的默认特征标名称。
        在 ask_to_store 中调用，避免重复的 if-elif 分支。
        """
        if source_type == 'irrep':
            return f"{source_name}_特征标"
        elif source_type == 'stored':
            return f"{source_name}_副本"
        elif source_type == 'spherical_harmonics':
            return f"{source_name}_轨道"
        elif source_type == 'polynomial':
            return f"{source_name}_多项式"
        else:  # manual 或其他
            return "自定义特征标"
    # ========================================================

    # ---------- 以下为原有方法（未修改，仅示意）----------
    def get_sorted_groups(self):
        """
        返回按分类和群阶次升序排序的点群名称列表。
        分类顺序由内部预定义的 category_order 列表决定。
        """
        # 定义分类显示顺序
        category_order = [
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
        
        # 收集所有群及其分类、阶次
        groups_info = []
        for name, table in self.character_tables.items():
            cat = table.get('category', '其他')
            order = table.get('order', 0)
            groups_info.append((name, cat, order))
        
        # 排序：先按分类索引（category_order 中的位置），再按阶次升序
        def sort_key(item):
            name, cat, order = item
            cat_index = category_order.index(cat) if cat in category_order else len(category_order)
            return (cat_index, order, name)   # 同阶次时按名字字母序
        
        groups_info.sort(key=sort_key)
        return [name for name, _, _ in groups_info]

    def verify_character_table(self, verbose=True):
        """验证当前群特征标表的正确性。
        参数:
            verbose: 是否打印详细验证过程，默认为 True。
        返回:
            bool: 是否通过所有验证项。
        """
        # ---------- 原有验证项 1~5（代码不变） ----------
        n_classes = len(self.class_sizes)
        n_irreps = len(self.irreps)

        if verbose:
            print(f"\n{'='*80}")
            print(f"验证 {self.current_group} 群特征标表")
            print(f"{'='*80}")
        
        n_classes = len(self.class_sizes)
        n_irreps = len(self.irreps)
        
        # 1. 检查共轭类数等于不可约表示数
        if verbose:
            print(f"1. 共轭类数 = 不可约表示数:")
            print(f"   共轭类数: {n_classes}")
            print(f"   不可约表示数: {n_irreps}")
            if n_classes == n_irreps:
                print(f"   ✓ 通过 (n_classes = n_irreps)")
            else:
                print(f"   ✗ 失败: 共轭类数 ({n_classes}) ≠ 不可约表示数 ({n_irreps})")
        
        # 2. 检查维数平方和等于群阶
        if verbose:
            print(f"\n2. 维数平方和 = 群阶:")
        sum_dim_squares = 0
        irrep_dims = []
        
        for irrep_name, character in self.irreps.items():
            dim = character[0]
            if isinstance(dim, complex):
                dim = dim.real
            dim = int(round(dim))
            irrep_dims.append(dim)
            sum_dim_squares += dim**2
        
        if verbose:
            print(f"   维数: {irrep_dims}")
            print(f"   维数平方和: Σ(dim²) = {sum_dim_squares}")
            print(f"   群阶: |G| = {self.group_order}")
            if sum_dim_squares == self.group_order:
                print(f"   ✓ 通过 (Σ(dim²) = |G|)")
            else:
                print(f"   ✗ 失败: Σ(dim²) = {sum_dim_squares} ≠ |G| = {self.group_order}")
        
        # 3. 检查每个不可约表示的维数整除群阶
        if verbose: print(f"\n3. 不可约表示的维数整除群阶:")
        all_divisible = True
        for i, dim in enumerate(irrep_dims):
            if self.group_order % dim == 0:
                if verbose: print(f"   {list(self.irreps.keys())[i]}: dim = {dim}, {self.group_order} / {dim} = {self.group_order // dim} ✓")
            else:
                if verbose: print(f"   {list(self.irreps.keys())[i]}: dim = {dim}, {self.group_order} 不能被 {dim} 整除 ✗")
                all_divisible = False
        
        if verbose:
            if all_divisible:
                print(f"   ✓ 所有不可约表示的维数都整除群阶")
            else:
                print(f"   ✗ 部分不可约表示的维数不能整除群阶")
        
        # 4. 检查行正交性（不可约表示之间的正交性）
        if verbose:
            print(f"\n4. 行正交性 (不可约表示之间的正交性):")
            print(f"   对于任意两个不同的不可约表示π和σ，有 ⟨χ_π, χ_σ⟩ = 0")
            print(f"   对于相同的不可约表示，有 ⟨χ_π, χ_π⟩ = |G|")
        
        row_orthogonal = True
        irreps_list = list(self.irreps.items())
        
        for i in range(n_irreps):
            for j in range(i, n_irreps):
                irrep1_name, char1 = irreps_list[i]
                irrep2_name, char2 = irreps_list[j]
                
                inner_product = 0
                for k in range(n_classes):
                    class_size = self.class_sizes[k]
                    chi1_val = char1[k]
                    chi2_val = char2[k]
                    
                    if isinstance(chi2_val, complex):
                        inner_product += class_size * chi1_val * chi2_val.conjugate()
                    else:
                        inner_product += class_size * chi1_val * chi2_val
                
                if isinstance(inner_product, complex):
                    inner_product = inner_product.real
                
                inner_product_rounded = round(inner_product)
                
                if i == j:
                    expected = self.group_order
                    if abs(inner_product_rounded - expected) < 1e-10:
                        if verbose: print(f"   ⟨{irrep1_name}, {irrep1_name}⟩ = {inner_product_rounded} ≈ {expected} ✓")
                    else:
                        if verbose: print(f"   ⟨{irrep1_name}, {irrep1_name}⟩ = {inner_product_rounded} ≠ {expected} ✗")
                        row_orthogonal = False
                else:
                    expected = 0
                    if abs(inner_product_rounded) < 1e-10:
                        if verbose: print(f"   ⟨{irrep1_name}, {irrep2_name}⟩ = {inner_product_rounded} ≈ 0 ✓")
                    else:
                        if verbose: print(f"   ⟨{irrep1_name}, {irrep2_name}⟩ = {inner_product_rounded} ≠ 0 ✗")
                        row_orthogonal = False
        
        if verbose:
            if row_orthogonal:
                print(f"   ✓ 所有行都满足正交性条件")
            else:
                print(f"   ✗ 部分行不满足正交性条件")
        
        # 5. 检查列正交性（共轭类之间的正交性）
        if verbose:
            print(f"\n5. 列正交性 (共轭类之间的正交性):")
            print("   对于任意两个不同的共轭类c和d，有 Σ_{π} χ_π(c) * conj(χ_π(d)) = 0")
            print("   对于相同的共轭类，有 Σ_{π} |χ_π(c)|² = |G| / |c|")
            
        column_orthogonal = True
        
        for i in range(n_classes):
            for j in range(i, n_classes):
                class_name_i = self.class_names[i]
                class_name_j = self.class_names[j]
                class_size_i = self.class_sizes[i]
                
                inner_product = 0
                for irrep_name, character in self.irreps.items():
                    chi_i = character[i]
                    chi_j = character[j]
                    
                    if isinstance(chi_j, complex):
                        inner_product += chi_i * chi_j.conjugate()
                    else:
                        inner_product += chi_i * chi_j
                
                if isinstance(inner_product, complex):
                    inner_product = inner_product.real
                
                inner_product_rounded = round(inner_product)
                
                if i == j:
                    expected = self.group_order / class_size_i
                    expected_rounded = round(expected)
                    
                    if abs(inner_product_rounded - expected_rounded) < 1e-10:
                        if verbose: print(f"   Σ|χ({class_name_i})|² = {inner_product_rounded} ≈ {expected_rounded} (|G|/|c| = {self.group_order}/{class_size_i}) ✓")
                    else:
                        if verbose: print(f"   Σ|χ({class_name_i})|² = {inner_product_rounded} ≠ {expected_rounded} (|G|/|c| = {self.group_order}/{class_size_i}) ✗")
                        column_orthogonal = False
                else:
                    expected = 0
                    if abs(inner_product_rounded) < 1e-10:
                        if verbose: print(f"   Σχ({class_name_i})·conj(χ({class_name_j})) = {inner_product_rounded} ≈ 0 ✓")
                    else:
                        if verbose: print(f"   Σχ({class_name_i})·conj(χ({class_name_j})) = {inner_product_rounded} ≠ 0 ✗")
                        column_orthogonal = False
        
        if verbose:
            if column_orthogonal:
                print(f"   ✓ 所有列都满足正交性条件")
            else:
                print(f"   ✗ 部分列不满足正交性条件")

        # ---------- 新增验证项 6~10 ----------
        all_tensor_passed = True
        all_sym2_passed = True
        all_sym3_passed = True
        all_alt2_passed = True   # 新增
        all_alt3_passed = True   # 新增
        vector_passed = True

        irreps_list = list(self.irreps.items())
        n_irreps = len(irreps_list)

        # ----- 6. 所有不可约表示两两张量积的分解 -----
        if verbose:
            print("\n6. 不可约表示张量积分解验证 (所有组合):")
        for i in range(n_irreps):
            name1, char1 = irreps_list[i]
            for j in range(i, n_irreps):
                name2, char2 = irreps_list[j]
                tensor_char = [a * b for a, b in zip(char1, char2)]
                try:
                    decomp = self.decompose(tensor_char)
                    if verbose:
                        print(f"   {name1} ⊗ {name2} = {self.format_decomposition(decomp)} ✓")
                except Exception as e:
                    if verbose:
                        print(f"   {name1} ⊗ {name2}: 分解失败 - {e} ✗")
                    all_tensor_passed = False

        # ----- 7. 所有不可约表示的 2 阶和 3 阶对称积分解 -----
        if verbose:
            print("\n7. 不可约表示对称积验证 (Sym² 和 Sym³):")
        for i in range(n_irreps):
            name, char = irreps_list[i]
            # Sym²
            try:
                sym2_char = self.symmetric_product_general(char, 2)
                decomp_sym2 = self.decompose(sym2_char)
                if verbose:
                    print(f"   Sym²({name}) = {self.format_decomposition(decomp_sym2)} ✓")
            except Exception as e:
                if verbose:
                    print(f"   Sym²({name}): 分解失败 - {e} ✗")
                all_sym2_passed = False
            # Sym³
            try:
                sym3_char = self.symmetric_product_general(char, 3)
                decomp_sym3 = self.decompose(sym3_char)
                if verbose:
                    print(f"   Sym³({name}) = {self.format_decomposition(decomp_sym3)} ✓")
            except Exception as e:
                if verbose:
                    print(f"   Sym³({name}): 分解失败 - {e} ✗")
                all_sym3_passed = False

        # ----- 8. 所有不可约表示的 2 阶和 3 阶反对称积分解 -----
        if verbose:
            print("\n8. 不可约表示反对称积验证 (Alt² 和 Alt³):")
        for i in range(n_irreps):
            name, char = irreps_list[i]
            # Alt²
            try:
                alt2_char = self.antisymmetric_product_general(char, 2)
                decomp_alt2 = self.decompose(alt2_char)
                if verbose:
                    print(f"   Alt²({name}) = {self.format_decomposition(decomp_alt2)} ✓")
            except Exception as e:
                if verbose:
                    print(f"   Alt²({name}): 分解失败 - {e} ✗")
                all_alt2_passed = False
            # Alt³
            try:
                alt3_char = self.antisymmetric_product_general(char, 3)
                decomp_alt3 = self.decompose(alt3_char)
                if verbose:
                    print(f"   Alt³({name}) = {self.format_decomposition(decomp_alt3)} ✓")
            except Exception as e:
                if verbose:
                    print(f"   Alt³({name}): 分解失败 - {e} ✗")
                all_alt3_passed = False

        # ----- 9. 向量表示 (vector_char) 分解验证 -----
        if verbose:
            print("\n9. 向量表示特征标分解验证:")
        if self.vector_char is not None:
            try:
                decomp_vector = self.decompose(self.vector_char)
                if verbose:
                    print(f"   vector_char = {self.vector_char}")
                    print(f"   分解 = {self.format_decomposition(decomp_vector)} ✓")
            except Exception as e:
                if verbose:
                    print(f"   vector_char 分解失败 - {e} ✗")
                vector_passed = False
        else:
            if verbose:
                print("   当前群未定义 vector_char，跳过此项验证。")
            # 视为通过（不强制要求）

        # ---------- 汇总所有验证结果 ----------
        if verbose:
            print(f"\n{'='*80}")
            print("验证总结:")
            print(f"{'='*80}")

        all_passed = (
            n_classes == n_irreps and
            sum_dim_squares == self.group_order and
            all_divisible and
            row_orthogonal and
            column_orthogonal and
            all_tensor_passed and
            all_sym2_passed and
            all_sym3_passed and
            all_alt2_passed and      # 新增
            all_alt3_passed and      # 新增
            vector_passed
        )

        if verbose:
            if all_passed:
                print(f"✓ {self.current_group} 群的特征标表通过所有验证!")
            else:
                print(f"✗ {self.current_group} 群的特征标表未通过全部验证")
                if not all_tensor_passed:
                    print("   - 部分张量积分解未通过")
                if not all_sym2_passed:
                    print("   - 部分 2 阶对称积分解未通过")
                if not all_sym3_passed:
                    print("   - 部分 3 阶对称积分解未通过")
                if not all_alt2_passed:    # 新增
                    print("   - 部分 2 阶反对称积分解未通过")
                if not all_alt3_passed:    # 新增
                    print("   - 部分 3 阶反对称积分解未通过")
                if not vector_passed and self.vector_char is not None:
                    print("   - 向量表示分解未通过")

        return all_passed
    
    def verify_all_tables(self, verbose=False):
        """
        验证所有已定义点群的特征标表。
        参数:
            verbose: 是否详细显示每个群的验证过程，默认为 False。
        返回:
            bool: 是否所有群都通过验证。
        """
        original_group = self.current_group
        total = len(self.character_tables)
        passed = 0
        failed_groups = []

        print(f"\n{'='*80}")
        print(f"开始验证所有 {total} 个点群的特征标表")
        print(f"{'='*80}")

        for i, group_name in enumerate(self.character_tables, 1):
            print(f"\n[{i}/{total}] 正在验证 {group_name} ...")
            try:
                self.set_group(group_name)
                result = self.verify_character_table(verbose=verbose)
                if result:
                    passed += 1
                    if not verbose:
                        print(f"✓ {group_name} 通过验证")
                else:
                    failed_groups.append(group_name)
                    if not verbose:
                        print(f"✗ {group_name} 未通过验证")
            except Exception as e:
                print(f"验证 {group_name} 时发生异常: {e}")
                failed_groups.append(group_name)

        # 恢复原有点群
        self.set_group(original_group)

        print(f"\n{'='*80}")
        print(f"总体验证结果: {passed}/{total} 个群通过验证。")
        if failed_groups:
            print(f"未通过验证的群: {', '.join(failed_groups)}")
        else:
            print("所有群的特征标表均通过验证！")
        print(f"{'='*80}")
        return passed == total

    def set_group(self, group_name):
        """设置当前使用的群"""
        if group_name not in self.character_tables:
            raise ValueError(f"未知的群: {group_name}。可用群: {list(self.character_tables.keys())}")
        
        self.current_group = group_name
        table = self.character_tables[group_name]
        
        self.irreps = table['irreps']
        self.class_sizes = table['class_sizes']
        self.class_names = table['class_names']
        self.group_order = table['order']
        self.class_cycles = table.get('class_cycles', None)
        
        # 读取向量表示的特征标（用于球谐函数/原子轨道）
        self.vector_char = table.get('vector_char', None)
        if self.vector_char is None:
            print(f"警告: 当前群 {self.current_group} 没有定义向量表示的特征标，球谐函数/原子轨道功能不可用")
        
        if 'special_notes' in table:
            self.special_notes = table['special_notes']
        else:
            self.special_notes = None
        
        if 'complex' in table:
            self.complex_table = table['complex']
        else:
            self.complex_table = False
    
    def get_available_groups(self):
        """获取所有可用的群"""
        return list(self.character_tables.keys())

    def get_character_at_power(self, character, n):
        """计算χ(g^n)的特征标"""
        if self.class_cycles is None:
            raise ValueError(f"当前群 {self.current_group} 没有定义循环子群信息")
        
        n_classes = len(self.class_sizes)
        power_char = [0] * n_classes
        
        for i in range(n_classes):
            order, cycle = self.class_cycles[i]
            n_mod = n % order
            class_index = cycle[n_mod]
            power_char[i] = character[class_index]
        
        return power_char
    
    def conjugate_classes_sn(self, n):
        """找出对称群 S_n 的所有共轭类（循环型）及其大小"""
        classes = []
        
        def generate_partitions(remaining, max_part, current):
            if remaining == 0:
                process_partition(current.copy())
                return
            
            for part in range(min(max_part, remaining), 0, -1):
                current.append(part)
                generate_partitions(remaining - part, part, current)
                current.pop()
        
        def process_partition(partition):
            cycle_counts = Counter(partition)
            m = len(partition)
            sign_factor = (-1) ** (n + m)
            
            denominator = 1
            for cycle_length, count in cycle_counts.items():
                denominator *= (cycle_length ** count) * math.factorial(count)
            
            class_size = math.factorial(n) // denominator
            
            desc_parts = []
            for cycle_length in sorted(cycle_counts.keys(), reverse=True):
                count = cycle_counts[cycle_length]
                if count == 1:
                    desc_parts.append(f"({cycle_length})")
                else:
                    desc_parts.append(f"({cycle_length})^{count}")
            
            cycle_type_desc = "·".join(desc_parts)
            
            classes.append((cycle_type_desc, partition.copy(), class_size, sign_factor))
        
        generate_partitions(n, n, [])
        classes.sort(key=lambda x: (len(x[1]), x[1]), reverse=True)
        return classes
    
    def print_conjugate_classes(self, n):
        """打印 S_n 的所有共轭类信息"""
        print(f"对称群 S_{n} 的共轭类：")
        print("=" * 60)
        print(f"{'循环型':<22} {'循环长度划分':<14} {'类大小':<7} {'符号(-1)^{n+m}'}")
        print("-" * 60)
        
        classes = self.conjugate_classes_sn(n)
        total = 0
        
        for cycle_type, partition, size, sign in classes:
            partition_str = " + ".join(map(str, partition))
            sign_str = f"(-1)^{n+len(partition)} = {sign}"
            print(f"{cycle_type:<25} {partition_str:<20} {size:<10} {sign_str}")
            total += size
        
        print("=" * 60)
        print(f"总置换数: {math.factorial(n)}")
        print(f"类和验证: {total} {'=' if total == math.factorial(n) else '!='} {math.factorial(n)}")
        
        return classes
    
    def find_class_by_cycle_type(self, n, cycle_lengths):
        """根据给定的循环长度查找对应的共轭类"""
        if sum(cycle_lengths) != n:
            print(f"错误: 循环长度之和 {sum(cycle_lengths)} 不等于 n={n}")
            return None
        
        sorted_lengths = sorted(cycle_lengths, reverse=True)
        cycle_counts = Counter(sorted_lengths)
        m = len(sorted_lengths)
        sign_factor = (-1) ** (n + m)
        
        denominator = 1
        for cycle_length, count in cycle_counts.items():
            denominator *= (cycle_length ** count) * math.factorial(count)
        
        class_size = math.factorial(n) // denominator
        
        desc_parts = []
        for cycle_length in sorted(cycle_counts.keys(), reverse=True):
            count = cycle_counts[cycle_length]
            if count == 1:
                desc_parts.append(f"({cycle_length})")
            else:
                desc_parts.append(f"({cycle_length})^{count}")
        
        cycle_type_desc = "·".join(desc_parts)
        
        return cycle_type_desc, sorted_lengths, class_size, sign_factor
    
    def symmetric_product_general(self, character, n):
        """
        使用置换群公式计算通用的对称积特征标
        支持 n=0 时返回平凡表示（全1向量）
        """
        if n == 0:
            return [1] * len(self.class_sizes)
        
        classes = self.conjugate_classes_sn(n)
        n_factorial = math.factorial(n)
        
        n_classes = len(self.class_sizes)
        sym_char = [0] * n_classes
        
        for _, partition, class_size, _ in classes:
            for class_idx in range(n_classes):
                product = 1.0
                for cycle_len in partition:
                    power_char = self.get_character_at_power(character, cycle_len)
                    product *= power_char[class_idx]
                sym_char[class_idx] += class_size * product
        
        sym_char = [x / n_factorial for x in sym_char]
        # ---------- 复数处理与整数舍入 ----------
        processed = []
        for x in sym_char:
            # 若为复数且虚部可忽略，视为实数
            if isinstance(x, complex) and abs(x.imag) < 1e-10:
                val = x.real
            else:
                val = x
            # 对实数尝试舍入到最近整数
            if isinstance(val, (int, float)):
                if abs(val - round(val)) < 1e-10:
                    val = round(val)
            processed.append(val)
        return processed
    
    def antisymmetric_product_general(self, character, n):
        """
        使用置换群公式计算通用的反对称积特征标
        支持 n=0 时返回平凡表示（全1向量）
        """
        if n == 0:
            return [1] * len(self.class_sizes)
        
        classes = self.conjugate_classes_sn(n)
        n_factorial = math.factorial(n)
        
        n_classes = len(self.class_sizes)
        antisym_char = [0] * n_classes
        
        for _, partition, class_size, sign in classes:
            for class_idx in range(n_classes):
                product = 1.0
                for cycle_len in partition:
                    power_char = self.get_character_at_power(character, cycle_len)
                    product *= power_char[class_idx]
                antisym_char[class_idx] += class_size * sign * product
        
        antisym_char = [x / n_factorial for x in antisym_char]
        # ---------- 复数处理与整数舍入 ----------
        processed = []
        for x in antisym_char:
            if isinstance(x, complex) and abs(x.imag) < 1e-10:
                val = x.real
            else:
                val = x
            if isinstance(val, (int, float)):
                if abs(val - round(val)) < 1e-10:
                    val = round(val)
            processed.append(val)
        return processed
    
    def symmetric_and_antisymmetric_products_general(self, character, n):
        """使用置换群公式计算通用的对称积和反对称积"""
        sym_char = self.symmetric_product_general(character, n)
        sym_decomp = self.decompose(sym_char)
        antisym_char = self.antisymmetric_product_general(character, n)
        antisym_decomp = self.decompose(antisym_char)
        return sym_char, sym_decomp, antisym_char, antisym_decomp
    
    def decompose(self, reducible_character):
        """将可约表示特征标分解为不可约表示的直和"""
        if len(reducible_character) != len(self.class_sizes):
            raise ValueError(f"当前群 {self.current_group} 的特征标向量长度应为 {len(self.class_sizes)}，但输入为 {len(reducible_character)}")
        
        decomposition = {}
        
        for irrep_name, irrep_character in self.irreps.items():
            inner_product = sum(
                self.class_sizes[i] * reducible_character[i] * np.conj(irrep_character[i])
                for i in range(len(self.class_sizes))
            )
            multiplicity = inner_product / self.group_order
            multiplicity_rounded = round(multiplicity.real)
            if abs(multiplicity.real - multiplicity_rounded) > 1e-10:
                if self.complex_table:
                    multiplicity_rounded = round(multiplicity.real)
                    if abs(multiplicity.real - multiplicity_rounded) > 1e-10:
                        raise ValueError(f"计算得到的{irrep_name}的多重度{multiplicity.real}不是整数，请检查输入特征标")
                else:
                    raise ValueError(f"计算得到的{irrep_name}的多重度{multiplicity.real}不是整数，请检查输入特征标")
            
            if multiplicity_rounded > 0:
                decomposition[irrep_name] = multiplicity_rounded
        
        return decomposition
    
    def format_decomposition(self, decomposition):
        """格式化分解结果为字符串"""
        if not decomposition:
            return "0 (不可约表示分解为空)"
        
        parts = []
        for irrep_name, multiplicity in sorted(decomposition.items()):
            if multiplicity == 1:
                parts.append(irrep_name)
            else:
                parts.append(f"{multiplicity}{irrep_name}")
        
        return " ⊕ ".join(parts)
    
    def print_character_table(self):
        """打印当前群的特征标表"""
        print(f"{self.current_group} 群特征标表:")
        print("=" * 80)
        header = f"{'不可约表示':<10} " + " ".join([f"{cls:<8}" for cls in self.class_names])
        print(header)
        print("-" * 80)
        for irrep_name, character in self.irreps.items():
            row = f"{irrep_name:<10} " + " ".join([f"{val:<8}" for val in character])
            print(row)
        print("-" * 80)
        print(f"群阶 h = {self.group_order}")
        print(f"类大小: {self.class_sizes}")
        if self.special_notes:
            print(f"注: {self.special_notes}")
    
    def tensor_product(self, char1, char2):
        """计算两个特征标的张量积特征标"""
        tensor_char = [a * b for a, b in zip(char1, char2)]
        decomposition = self.decompose(tensor_char)
        return tensor_char, decomposition
    
    def direct_sum(self, char1, char2):
        """计算两个特征标的直和特征标"""
        direct_sum_char = [a + b for a, b in zip(char1, char2)]
        decomposition = self.decompose(direct_sum_char)
        return direct_sum_char, decomposition
    
    def get_high_power_character(self, character, power):
        """计算高幂次的特征标χ(g^power)"""
        return self.get_character_at_power(character, power)
    
    # ============== 球谐函数/原子轨道相关（未修改）==============
    def parse_orbital_input(self, input_str):
        """
        解析轨道输入，返回角量子数 l
        支持输入数字（如 0,1,2,...）或轨道字母（s,p,d,f,g,h,i,j,k,l,m,n,o）
        """
        input_str = input_str.strip().lower()
        # 尝试转换为整数
        try:
            l = int(input_str)
            if l < 0:
                raise ValueError(f"角量子数不能为负数: {l}")
            return l
        except ValueError:
            # 轨道字母映射
            orbital_map = {
                's': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4, 'h': 5, 'i': 6,
                'j': 7, 'k': 8, 'l': 9, 'm': 10, 'n': 11, 'o': 12
            }
            if input_str in orbital_map:
                return orbital_map[input_str]
            else:
                raise ValueError(f"无效的轨道输入: '{input_str}'，应为数字或轨道字母(s,p,d,f,g,h,i,j,k,l,m,n,o)")
    
    def harmonic_character(self, l):
        """
        计算角量子数为 l 的球谐函数表示（调和多项式空间 H_l）的特征标向量
        使用递归公式：χ_{H_l} = χ_{Sym^l V} - χ_{Sym^{l-2} V}
        """
        if self.vector_char is None:
            raise ValueError(f"当前群 {self.current_group} 没有定义向量表示的特征标，无法生成球谐函数表示")
        
        if l < 0:
            raise ValueError("角量子数不能为负数")
        
        if l == 0:
            return [1] * len(self.class_sizes)
        if l == 1:
            return self.vector_char.copy()
        
        # 对于 l >= 2，使用递归公式
        sym_l = self.symmetric_product_general(self.vector_char, l)
        if l - 2 == 0:
            sym_lm2 = [1] * len(self.class_sizes)
        else:
            sym_lm2 = self.symmetric_product_general(self.vector_char, l - 2)
        
        return [sym_l[i] - sym_lm2[i] for i in range(len(self.class_sizes))]
    
    # ============== 特征标存储和调用（仅 ask_to_store 修改）==============
    def store_character(self, name, character, description=""):
        """存储特征标"""
        if self.current_group not in self.stored_characters:
            self.stored_characters[self.current_group] = {}
        
        self.stored_characters[self.current_group][name] = {
            'character': character,
            'description': description
        }
        
        self.save_characters()
        print(f"特征标 '{name}' 已存储")
    
    def get_stored_character(self, name):
        """获取存储的特征标"""
        if self.current_group in self.stored_characters and name in self.stored_characters[self.current_group]:
            char_info = self.stored_characters[self.current_group][name]
            return char_info['character'], char_info['description']
        else:
            return None, ""
    
    def list_stored_characters(self):
        """列出当前群所有存储的特征标"""
        if self.current_group in self.stored_characters:
            return list(self.stored_characters[self.current_group].keys())
        else:
            return []
    
    def delete_stored_character(self, name):
        """删除存储的特征标"""
        if self.current_group in self.stored_characters and name in self.stored_characters[self.current_group]:
            del self.stored_characters[self.current_group][name]
            self.save_characters()
            print(f"特征标 '{name}' 已删除")
        else:
            print(f"特征标 '{name}' 不存在")
    
    def save_characters(self):
        """保存特征标到文件"""
        try:
            serializable = {}
            for group_name, chars in self.stored_characters.items():
                serializable[group_name] = {}
                for char_name, char_info in chars.items():
                    serializable[group_name][char_name] = {
                        'character': [float(x) if isinstance(x, (int, float, np.integer, np.floating)) else complex(x) for x in char_info['character']],
                        'description': char_info['description']
                    }
            
            with open('stored_characters.json', 'w') as f:
                json.dump(serializable, f, default=str)
        except Exception as e:
            print(f"保存特征标时出错: {e}")
    
    def load_characters(self):
        """从文件加载特征标"""
        try:
            with open('stored_characters.json', 'r') as f:
                loaded = json.load(f)
                
            self.stored_characters = {}
            for group_name, chars in loaded.items():
                self.stored_characters[group_name] = {}
                for char_name, char_info in chars.items():
                    character = []
                    for val in char_info['character']:
                        if isinstance(val, str) and 'j' in val:
                            character.append(complex(val))
                        else:
                            character.append(float(val))
                    
                    self.stored_characters[group_name][char_name] = {
                        'character': character,
                        'description': char_info['description']
                    }
                    
            print(f"已加载 {sum(len(chars) for chars in self.stored_characters.values())} 个存储的特征标")
        except FileNotFoundError:
            print("未找到存储的特征标文件，将创建新文件")
        except Exception as e:
            print(f"加载特征标时出错: {e}")
    
    # ---------- 修改：input_character 仅调整菜单提示 ----------
    def input_character(self, prompt="请选择输入方式"):
        """
        交互式输入特征标
        选项 4：球谐函数/原子轨道
        选项 5：多项式 (Polynomial)
        """
        print(f"\n{prompt}:")
        print("  1. 手动输入特征标")
        print("  2. 使用存储的特征标")
        print("  3. 使用不可约表示")
        print("  4. 使用球谐函数/原子轨道")
        print("  5. 使用多项式 (Polynomial)")   # [MODIFIED] 菜单文字简化

        while True:
            choice = input("请选择 (1-5): ").strip()
            if choice == "1":
                # 手动输入（原代码不变）
                print(f"\n请输入 {self.current_group} 群的特征标向量")
                print(f"共轭类顺序: {self.class_names}")
                print(f"类大小: {self.class_sizes}")
                print(f"需要 {len(self.class_sizes)} 个数值")
                
                while True:
                    user_input = input("用空格分隔 (或输入 'q' 取消): ").strip()
                    if user_input.lower() == 'q':
                        return None, None, None
                    
                    try:
                        values = user_input.split()
                        char_vector = []
                        for val in values:
                            if 'j' in val or 'i' in val:
                                val = val.replace('i', 'j')
                                char_vector.append(complex(val))
                            else:
                                char_vector.append(float(val))
                        
                        if len(char_vector) != len(self.class_sizes):
                            print(f"错误: 需要 {len(self.class_sizes)} 个数值，但输入了 {len(char_vector)} 个")
                            continue
                        
                        return char_vector, 'manual', "手动输入"
                        
                    except ValueError:
                        print("错误: 请输入有效的数字，用空格分隔")
            
            elif choice == "2":
                # 使用存储的特征标（原代码不变）
                stored_names = self.list_stored_characters()
                if not stored_names:
                    print("当前群没有存储的特征标")
                    continue
                
                print("\n存储的特征标:")
                for i, name in enumerate(stored_names, 1):
                    char, desc = self.get_stored_character(name)
                    if desc:
                        print(f"  {i}. {name}: {desc}")
                    else:
                        print(f"  {i}. {name}")
                
                while True:
                    try:
                        idx_input = input("选择特征标 (输入序号或名称，或输入 'q' 取消): ").strip()
                        if idx_input.lower() == 'q':
                            return None, None, None
                        
                        if idx_input.isdigit():
                            idx = int(idx_input) - 1
                            if 0 <= idx < len(stored_names):
                                name = stored_names[idx]
                                char, desc = self.get_stored_character(name)
                                print(f"已选择特征标 '{name}'")
                                return char, 'stored', name
                            else:
                                print(f"序号超出范围 (1-{len(stored_names)})")
                        else:
                            if idx_input in stored_names:
                                char, desc = self.get_stored_character(idx_input)
                                print(f"已选择特征标 '{idx_input}'")
                                return char, 'stored', idx_input
                            else:
                                print(f"特征标 '{idx_input}' 不存在")
                                
                    except ValueError:
                        print("请输入有效的序号或名称")
            
            elif choice == "3":
                # 使用不可约表示（原代码不变）
                print(f"\n{self.current_group} 群的不可约表示:")
                irreps_list = list(self.irreps.keys())
                for i, irrep in enumerate(irreps_list, 1):
                    char = self.irreps[irrep]
                    print(f"  {i}. {irrep}: {char}")
                
                while True:
                    try:
                        idx_input = input("选择不可约表示 (输入序号或名称，或输入 'q' 取消): ").strip()
                        if idx_input.lower() == 'q':
                            return None, None, None
                        
                        if idx_input.isdigit():
                            idx = int(idx_input) - 1
                            if 0 <= idx < len(irreps_list):
                                name = irreps_list[idx]
                                char = self.irreps[name]
                                print(f"已选择不可约表示 '{name}'")
                                return char, 'irrep', name
                            else:
                                print(f"序号超出范围 (1-{len(irreps_list)})")
                        else:
                            if idx_input in self.irreps:
                                char = self.irreps[idx_input]
                                print(f"已选择不可约表示 '{idx_input}'")
                                return char, 'irrep', idx_input
                            else:
                                print(f"不可约表示 '{idx_input}' 不存在")
                                
                    except ValueError:
                        print("请输入有效的序号或名称")
            
            elif choice == "4":
                # 球谐函数/原子轨道（原代码不变）
                if self.vector_char is None:
                    print(f"错误: 当前群 {self.current_group} 没有定义向量表示的特征标，无法使用此功能")
                    continue
                
                print("\n球谐函数/原子轨道表示:")
                print("  支持输入角量子数 l (0,1,2,...) 或轨道字母 (s,p,d,f,g,h,i,j,k,l,m,n,o)")
                while True:
                    orbital_input = input("请输入 l 或轨道字母 (如 '2', 'd', 'p' 等，输入 'q' 取消): ").strip()
                    if orbital_input.lower() == 'q':
                        return None, None, None
                    
                    try:
                        l = self.parse_orbital_input(orbital_input)
                        if l > 12:
                            confirm = input(f"l = {l} 较大，对称积计算可能较慢，是否继续? (y/n): ").strip().lower()
                            if confirm not in ('y', 'yes'):
                                continue
                        char_vector = self.harmonic_character(l)
                        if orbital_input.lower() in ['s','p','d','f','g','h','i','j','k','l','m','n','o']:
                            name = orbital_input.lower()
                        else:
                            name = f"l={l}"
                        print(f"已生成球谐函数表示: {name}，特征标: {char_vector}")
                        return char_vector, 'spherical_harmonics', name
                    except ValueError as e:
                        print(f"错误: {e}")
                    except Exception as e:
                        print(f"生成特征标时出错: {e}")
            
            elif choice == "5":
                # 多项式（向量表示的对称幂）
                if self.vector_char is None:
                    print(f"错误: 当前群 {self.current_group} 没有定义向量表示的特征标，无法使用此功能")
                    continue
                
                print("\n多项式 (Polynomial):")   # [MODIFIED] 菜单文字
                print("  输入齐次多项式的次数 n (n ≥ 0 的整数)，计算多项式表示的特征标")
                while True:
                    n_input = input("请输入 n (如 '2', '3' 等，输入 'q' 取消): ").strip()
                    if n_input.lower() == 'q':
                        return None, None, None
                    
                    try:
                        n = int(n_input)
                        if n < 0:
                            print("错误: n 必须 ≥ 0")
                            continue
                        if n > 10:
                            confirm = input(f"n = {n} 较大，对称积计算可能较慢，是否继续? (y/n): ").strip().lower()
                            if confirm not in ('y', 'yes'):
                                continue
                        
                        char_vector = self.symmetric_product_general(self.vector_char, n)
                        name = f"P_{n}"          # [MODIFIED] 固定格式 P_n
                        print(f"已生成多项式表示: {name}，特征标: {char_vector}")
                        return char_vector, 'polynomial', name
                    except ValueError:
                        print("错误: 请输入有效的整数")
                    except Exception as e:
                        print(f"生成特征标时出错: {e}")
            else:
                print("无效选择，请重试 (1-5)")

    # ---------- 修改：ask_to_store 使用统一方法 get_default_name ----------
    def ask_to_store(self, character, default_name="", source_type="", source_name=""):
        """询问是否存储特征标（使用统一的默认名称生成方法）"""
        store = input(f"\n是否存储此特征标? (y/n, 默认为n): ").strip().lower()
        if store in ('y', 'yes'):
            if default_name:
                name = input(f"输入特征标名称 (默认为 '{default_name}'): ").strip()
                if not name:
                    name = default_name
            else:
                # [MODIFIED] 调用统一方法生成默认名称
                default_name = self.get_default_name(source_type, source_name)
                name = input(f"输入特征标名称 (默认为 '{default_name}'): ").strip()
                if not name:
                    name = default_name
            
            stored_names = self.list_stored_characters()
            if name in stored_names:
                overwrite = input(f"特征标 '{name}' 已存在，是否覆盖? (y/n): ").strip().lower()
                if overwrite not in ('y', 'yes'):
                    return False
            
            description = input("输入特征标描述 (可选): ").strip()
            self.store_character(name, character, description)
            return True
        return False


# ---------- 修改：group_operations_loop 使用统一方法 get_source_display ----------
def group_operations_loop(decomposer, group_name, groups):
    """处理特定点群的操作循环（使用统一的来源显示方法）"""
    while True:
        print("\n" + "=" * 80)
        print(f"特征标计算器 (当前群: {group_name})")
        print(f"共轭类顺序: {decomposer.class_names}")
        print(f"类大小: {decomposer.class_sizes}")
        print("=" * 80)
        
        print("\n操作菜单:")
        print("  1. 特征标分解")
        print("  2. 张量积")
        print("  3. 对称积和反对称积")
        print("  4. 表示的直和")
        print("  5. 操作幂次的特征标")
        print("  6. 管理存储的特征标")
        print("  7. 查看特征标表")
        print("  8. 验证特征标表")
        print("  9. 置换群Sn的共轭类")
        print("  10. 返回点群选择")
        
        try:
            choice = input("请选择 (1-10): ").strip()
            
            if choice == "1":
                char_vector, source, char_name = decomposer.input_character("特征标分解")
                if char_vector is None:
                    print("操作取消")
                    continue
                
                decomposition = decomposer.decompose(char_vector)
                decomposition_str = decomposer.format_decomposition(decomposition)
                
                # [MODIFIED] 统一调用 get_source_display
                if source == 'irrep':
                    print(f"\n不可约表示 '{char_name}':")
                    print(f"特征标: {char_vector}")
                    print(f"分解: {char_name}")
                else:
                    source_display = decomposer.get_source_display(source, char_name)
                    print(f"\n{source_display} 的特征标分解:")
                    print(f"特征标: {char_vector}")
                    print(f"不可约表示分解: {decomposition_str}")
                
                if source != 'irrep':
                    decomposer.ask_to_store(char_vector, default_name=char_name,
                                          source_type=source, source_name=char_name)
                elif source == 'irrep':
                    store = input(f"\n是否存储不可约表示 '{char_name}' 的特征标? (y/n): ").strip().lower()
                    if store in ('y', 'yes'):
                        default_name = f"{char_name}_特征标"
                        decomposer.ask_to_store(char_vector, default_name=default_name,
                                              source_type=source, source_name=char_name)
            
            elif choice == "2":
                # 张量积计算
                print("\n选择第一个特征标:")
                char1, source1, name1 = decomposer.input_character()
                if char1 is None: continue
                print("\n选择第二个特征标:")
                char2, source2, name2 = decomposer.input_character()
                if char2 is None: continue
                
                tensor_char, decomposition = decomposer.tensor_product(char1, char2)
                decomposition_str = decomposer.format_decomposition(decomposition)
                
                # [MODIFIED] 统一调用 get_source_display
                source1_display = decomposer.get_source_display(source1, name1)
                source2_display = decomposer.get_source_display(source2, name2)
                
                print(f"\n{source1_display} ⊗ {source2_display}:")
                print(f"特征标: {tensor_char}")
                print(f"分解: {decomposition_str}")
                
                store = input(f"\n是否存储张量积结果? (y/n): ").strip().lower()
                if store == 'y' or store == 'yes':
                    default_name = ""
                    if name1 and name2:
                        default_name = f"{name1}⊗{name2}"
                    elif name1:
                        default_name = f"{name1}⊗自定义"
                    elif name2:
                        default_name = f"自定义⊗{name2}"
                    else:
                        default_name = "张量积"
                    
                    result_name = input(f"输入结果名称 (默认为 '{default_name}'): ").strip()
                    if not result_name:
                        result_name = default_name
                    
                    description = f"{source1_display} ⊗ {source2_display} 的分解: {decomposition_str}"
                    decomposer.store_character(result_name, tensor_char, description)
            
            elif choice == "3":
                # 对称积和反对称积
                print("\n计算对称积和反对称积")
                try:
                    n = int(input("请输入对称/反对称积的阶数:").strip())
                    if n < 0: continue
                except ValueError:
                    print("请输入有效的整数")
                    continue
                
                char, source, name = decomposer.input_character("选择特征标")
                if char is None: continue
                
                sym_char, sym_decomp, antisym_char, antisym_decomp = \
                    decomposer.symmetric_and_antisymmetric_products_general(char, n)
                sym_str = decomposer.format_decomposition(sym_decomp)
                antisym_str = decomposer.format_decomposition(antisym_decomp)
                
                # [MODIFIED] 统一调用 get_source_display
                source_display = decomposer.get_source_display(source, name)
                
                print(f"\n{source_display} 的对称 {n} 次积 [Sym^{n}]:")
                print(f"特征标: {sym_char}")
                print(f"分解: {sym_str}")
                print(f"\n{source_display} 的反对称 {n} 次积 [Alt^{n}]:")
                print(f"特征标: {antisym_char}")
                print(f"分解: {antisym_str}")
                
                store = input(f"\n是否存储计算结果? (y/n): ").strip().lower()
                if store == 'y' or store == 'yes':
                    sym_name_input = input(f"输入对称积名称 (默认为 '{name}_Sym{n}'): ").strip()
                    sym_name = sym_name_input if sym_name_input else f"{name}_Sym{n}"
                    sym_desc = f"{source_display} 的对称积 (n={n}), 分解: {sym_str}"
                    decomposer.store_character(sym_name, sym_char, sym_desc)
                    
                    antisym_name_input = input(f"输入反对称积名称 (默认为 '{name}_Alt{n}'): ").strip()
                    antisym_name = antisym_name_input if antisym_name_input else f"{name}_Alt{n}"
                    antisym_desc = f"{source_display} 的反对称积 (n={n}), 分解: {antisym_str}"
                    decomposer.store_character(antisym_name, antisym_char, antisym_desc)
            
            elif choice == "4":
                # 表示的直和
                print("\n计算表示的直和")
                print("选择第一个特征标:")
                char1, source1, name1 = decomposer.input_character()
                if char1 is None: continue
                print("选择第二个特征标:")
                char2, source2, name2 = decomposer.input_character()
                if char2 is None: continue
                
                direct_sum_char, decomposition = decomposer.direct_sum(char1, char2)
                decomposition_str = decomposer.format_decomposition(decomposition)
                
                # [MODIFIED] 统一调用 get_source_display
                source1_display = decomposer.get_source_display(source1, name1)
                source2_display = decomposer.get_source_display(source2, name2)
                
                print(f"\n{source1_display} ⊕ {source2_display}:")
                print(f"特征标: {direct_sum_char}")
                print(f"分解: {decomposition_str}")
                
                store = input(f"\n是否存储直和结果? (y/n): ").strip().lower()
                if store == 'y' or store == 'yes':
                    default_name = ""
                    if name1 and name2:
                        default_name = f"{name1}⊕{name2}"
                    elif name1:
                        default_name = f"{name1}⊕自定义"
                    elif name2:
                        default_name = f"自定义⊕{name2}"
                    else:
                        default_name = "直和"
                    
                    result_name = input(f"输入结果名称 (默认为 '{default_name}'): ").strip()
                    if not result_name:
                        result_name = default_name
                    
                    description = f"{source1_display} ⊕ {source2_display} 的分解: {decomposition_str}"
                    decomposer.store_character(result_name, direct_sum_char, description)
            
            elif choice == "5":
                # 高幂次特征标计算
                print("\n操作幂次的特征标计算")
                print("计算χ(g^n)，其中n可以是任意整数")
                char, source, name = decomposer.input_character("选择特征标")
                if char is None: continue
                
                while True:
                    try:
                        power = int(input("输入幂次n (整数): ").strip())
                        break
                    except ValueError:
                        print("请输入有效的整数")
                
                high_power_char = decomposer.get_high_power_character(char, power)
                decomposition = decomposer.decompose(high_power_char)
                decomposition_str = decomposer.format_decomposition(decomposition)
                
                # [MODIFIED] 统一调用 get_source_display
                source_display = decomposer.get_source_display(source, name)
                
                print(f"\n{source_display} 的 {power} 次幂特征标:")
                print(f"χ(g^{power}): {high_power_char}")
                print(f"分解: {decomposition_str}")
                
                store = input(f"\n是否存储高幂次特征标结果? (y/n): ").strip().lower()
                if store == 'y' or store == 'yes':
                    default_name = f"{name}_幂{power}" if name else f"幂{power}"
                    result_name = input(f"输入结果名称 (默认为 '{default_name}'): ").strip()
                    if not result_name:
                        result_name = default_name
                    
                    description = f"{source_display} 的 {power} 次幂特征标，分解: {decomposition_str}"
                    decomposer.store_character(result_name, high_power_char, description)
            
            # ---------- 选项6-10 原代码不变，此处省略 ----------
            elif choice == "6":
                # 管理存储的特征标（原代码不变）
                while True:
                    print("\n特征标管理:")
                    print("  1. 查看所有存储的特征标")
                    print("  2. 删除特征标")
                    print("  3. 返回操作菜单")
                    
                    mgmt_choice = input("请选择 (1-3): ").strip()
                    
                    if mgmt_choice == "1":
                        stored_names = decomposer.list_stored_characters()
                        if not stored_names:
                            print("当前群没有存储的特征标")
                        else:
                            print(f"\n{group_name} 群存储的特征标 ({len(stored_names)} 个):")
                            for i, name in enumerate(stored_names, 1):
                                char, desc = decomposer.get_stored_character(name)
                                print(f"  {i}. {name}")
                                if desc:
                                    print(f"     描述: {desc}")
                                print(f"     特征标: {char}")
                                print()
                    
                    elif mgmt_choice == "2":
                        stored_names = decomposer.list_stored_characters()
                        if not stored_names:
                            print("当前群没有存储的特征标")
                            continue
                        
                        print("\n存储的特征标:")
                        for i, name in enumerate(stored_names, 1):
                            print(f"  {i}. {name}")
                        
                        while True:
                            name_input = input("输入要删除的特征标名称 (或序号，输入 'q' 取消): ").strip()
                            if name_input.lower() == 'q':
                                break
                            
                            if name_input.isdigit():
                                idx = int(name_input) - 1
                                if 0 <= idx < len(stored_names):
                                    name = stored_names[idx]
                                    confirm = input(f"确认删除特征标 '{name}'? (y/n): ").strip().lower()
                                    if confirm == 'y' or confirm == 'yes':
                                        decomposer.delete_stored_character(name)
                                    break
                                else:
                                    print(f"序号超出范围 (1-{len(stored_names)})")
                            else:
                                if name_input in stored_names:
                                    confirm = input(f"确认删除特征标 '{name_input}'? (y/n): ").strip().lower()
                                    if confirm == 'y' or confirm == 'yes':
                                        decomposer.delete_stored_character(name_input)
                                    break
                                else:
                                    print(f"特征标 '{name_input}' 不存在")
                    
                    elif mgmt_choice == "3":
                        break
                    else:
                        print("无效选择")
                
            elif choice == "7":
                decomposer.print_character_table()
            elif choice == "8":
                decomposer.verify_character_table()
            elif choice == "9":
                print("\n查看置换群 S_n 的共轭类")
                while True:
                    try:
                        n_input = input("请输入 n (置换群S_n的阶，输入 'q' 返回): ").strip()
                        if n_input.lower() == 'q':
                            break
                        n = int(n_input)
                        if n < 1:
                            print("n必须大于等于1")
                            continue
                        classes = decomposer.print_conjugate_classes(n)
                        analyze = input(f"\n是否使用该置换群计算特征标的对称积和反对称积(也就是计算{n}重对称积和反对称积)? (y/n): ").strip().lower()
                        if analyze == 'y':
                            char, source, name = decomposer.input_character("选择特征标进行分析")
                            if char is not None:
                                print(f"\n使用 S_{n} 的共轭类计算对称积:")
                                sym_char = decomposer.symmetric_product_general(char, n)
                                sym_decomp = decomposer.decompose(sym_char)
                                sym_str = decomposer.format_decomposition(sym_decomp)
                                print(f"对称 {n} 次积 [Sym^{n}] 特征标: {sym_char}")
                                print(f"分解: {sym_str}")
                                
                                print(f"\n使用 S_{n} 的共轭类计算反对称积:")
                                antisym_char = decomposer.antisymmetric_product_general(char, n)
                                antisym_decomp = decomposer.decompose(antisym_char)
                                antisym_str = decomposer.format_decomposition(antisym_decomp)
                                print(f"反对称 {n} 次积 [Alt^{n}] 特征标: {antisym_char}")
                                print(f"分解: {antisym_str}")
                        break
                    except ValueError:
                        print("请输入有效的整数")
            elif choice == "10":
                print("返回点群选择")
                break
            else:
                print("无效选择，请重试")
        except Exception as e:
            print(f"发生错误: {e}")

def main():
    # 显示程序标题和版权信息
    print("=" * 80)
    print("特征标计算器 (Character Table Decomposer)")
    print(f"版本: {__version__}")
    print(f"作者: {__author__}")
    print(f"许可证: {__license__}")
    print(f"版权: {__copyright__}")
    print("=" * 80)
    print("此程序是自由软件，遵循 MIT 许可证。")
    print("您可以在遵守许可证条款的前提下自由使用、修改和分发。")
    print("=" * 80)

    decomposer = CharacterTableDecomposer()
    
    while True:
        print("=" * 80)
        print("特征标计算器")
        print("=" * 80)
        
        print("\n可用的点群:")
        # 获取排序后的群列表
        sorted_groups = decomposer.get_sorted_groups()
        
        # 按分类分组显示
        category_order = [  # 与排序方法中的顺序保持一致
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

        # 初始化分类字典
        categorized = {cat: [] for cat in category_order}
        # 将群放入对应分类
        for name in sorted_groups:
            cat = decomposer.character_tables[name].get('category', '其他')
            if cat in categorized:
                categorized[cat].append(name)
            else:
                if '其他' not in categorized:
                    categorized['其他'] = []
                categorized['其他'].append(name)

        # 显示分类和群名，并构建全局序号映射
        display_order_groups = []
        idx = 1
        for cat in category_order:
            if categorized[cat]:
                print(f"\n  {cat}:")
                for name in categorized[cat]:
                    print(f"    {idx}. {name}")
                    display_order_groups.append(name)
                    idx += 1
        
        # 处理“其他”分类
        if '其他' in categorized and categorized['其他']:
            print(f"\n  其他:")
            for name in categorized['其他']:
                print(f"    {idx}. {name}")
                display_order_groups.append(name)
                idx += 1
        
        print("  V. 验证所有特征标表")
        print("  0. 退出程序")
        
        try:
            choice = input(f"\n选择点群 (1-{len(display_order_groups)}, 0退出, 默认为1-O_h): ").strip().lower()
            
            if choice == "0":
                print("感谢使用特征标计算器，再见！")
                break
            
            if choice == "v":
                decomposer.verify_all_tables(verbose=False)   # 简洁模式，不打印细节
                continue

            if choice == "":
                group_name = 'O_h'   # 保持默认
            else:
                idx_choice = int(choice) - 1
                if 0 <= idx_choice < len(display_order_groups):
                    group_name = display_order_groups[idx_choice]
                else:
                    print(f"请输入0-{len(display_order_groups)}之间的数字")
                    continue
            
            decomposer.set_group(group_name)
            print(f"\n已选择 {group_name} 群")
            
            print("\n" + "=" * 80)
            decomposer.print_character_table()
            
            group_operations_loop(decomposer, group_name, sorted_groups)
            
        except ValueError as e:
            print(f"输入无效: {e}")
        except Exception as e:
            print(f"发生错误: {e}")

if __name__ == "__main__":

    main()
