"""
Core mathematical operations on character tables
Imports: constants, character_table_database
"""

from typing import Dict, List, Tuple, Union, Optional
from collections import Counter
import math
import numpy as np

from constants import TOLERANCE, MAX_ORBITAL_L
from character_table_database import CharacterTable


class CharacterCalculator:
    """Pure mathematical operations without I/O"""
    
    def __init__(self, table: CharacterTable):
        """Initialize calculator with a character table"""
        self.table = table
        self.irreps = table.irreps
        self.class_sizes = table.class_sizes
        self.class_names = table.class_names
        self.group_order = table.order
        self.class_cycles = table.class_cycles
        self.vector_char = table.vector_char
        self.is_complex = table.is_complex
    
    # ==================== Utility Methods ====================
    
    @staticmethod
    def to_real_if_possible(value: Union[int, float, complex], 
                           tolerance: float = TOLERANCE) -> Union[int, float]:
        """Convert complex to real if imaginary part is negligible"""
        if isinstance(value, complex):
            if abs(value.imag) < tolerance:
                return value.real
        return value
    
    @staticmethod
    def normalize_number(value: Union[int, float, complex],
                        tolerance: float = TOLERANCE) -> Union[int, float]:
        """Convert to real, then round to nearest integer if close"""
        real_val = CharacterCalculator.to_real_if_possible(value, tolerance)
        if isinstance(real_val, (int, float)):
            rounded = round(real_val)
            if abs(real_val - rounded) < tolerance:
                return rounded
        return real_val
    
    # ==================== Spherical Harmonics / Atomic Orbitals ====================
    
    @staticmethod
    def parse_orbital_input(input_str: str) -> int:
        """
        Parse orbital input, return angular quantum number l
        
        Supports:
        - Numbers: '0', '1', '2', etc.
        - Orbital letters: 's', 'p', 'd', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o'
        
        Args:
            input_str: User input
            
        Returns:
            Angular quantum number l
            
        Raises:
            ValueError: If input is invalid
        """
        input_str = input_str.strip().lower()
        
        # Try as integer
        try:
            l = int(input_str)
            if l < 0:
                raise ValueError(f"Angular quantum number cannot be negative: {l}")
            return l
        except ValueError:
            pass
        
        # Try as orbital letter
        orbital_map = {
            's': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4, 'h': 5, 'i': 6,
            'j': 7, 'k': 8, 'l': 9, 'm': 10, 'n': 11, 'o': 12
        }
        
        if input_str in orbital_map:
            return orbital_map[input_str]
        
        raise ValueError(
            f"Invalid orbital input: '{input_str}'. "
            f"Use number (0,1,2,...) or letter (s,p,d,f,g,h,i,j,k,l,m,n,o)"
        )
    
    def harmonic_character(self, l: int) -> List[Union[int, float]]:
        """
        Calculate spherical harmonic (atomic orbital) character
        
        For angular quantum number l, the character is computed using:
        χ_{H_l} = χ_{Sym^l V} - χ_{Sym^{l-2} V}
        
        where V is the 3D vector representation
        
        Args:
            l: Angular quantum number
            
        Returns:
            Character vector for harmonic of order l
            
        Raises:
            ValueError: If vector_char not defined or l < 0
        """
        if self.vector_char is None:
            raise ValueError(
                f"Vector representation not defined for {self.table.name}. "
                f"Cannot compute spherical harmonics."
            )
        
        if l < 0:
            raise ValueError("Angular quantum number cannot be negative")
        
        if l == 0:
            # Constant (s orbital)
            return [1] * len(self.class_sizes)
        
        if l == 1:
            # Vector representation (p orbitals)
            return self.vector_char.copy()
        
        # For l >= 2: use recursive formula
        sym_l = self.symmetric_product_general(self.vector_char, l)
        
        if l == 2:
            sym_lm2 = [1] * len(self.class_sizes)
        else:
            sym_lm2 = self.symmetric_product_general(self.vector_char, l - 2)
        
        return [sym_l[i] - sym_lm2[i] for i in range(len(self.class_sizes))]
    
    def polynomial_character(self, n: int) -> List[Union[int, float]]:
        """
        Calculate polynomial (symmetric power) character
        
        Represents the space of n-th degree homogeneous polynomials
        in 3 variables: Sym^n(V)
        
        Args:
            n: Degree of polynomial
            
        Returns:
            Character vector for Sym^n(V)
            
        Raises:
            ValueError: If vector_char not defined
        """
        if self.vector_char is None:
            raise ValueError(
                f"Vector representation not defined for {self.table.name}. "
                f"Cannot compute polynomials."
            )
        
        return self.symmetric_product_general(self.vector_char, n)
    
    # ==================== Mathematical Operations ====================
    
    def decompose(self, reducible_character: List[Union[int, float, complex]]) -> Dict[str, int]:
        """
        Decompose a reducible representation into irreducible components
        
        Uses the formula: n_i = (1/|G|) Σ_c |G_c| * χ(c) * χ̄_i(c)
        
        Args:
            reducible_character: Character vector of reducible representation
            
        Returns:
            Dict mapping irrep name to multiplicity
            
        Raises:
            ValueError: If character length doesn't match group structure
        """
        if len(reducible_character) != len(self.class_sizes):
            raise ValueError(
                f"Expected {len(self.class_sizes)} character values for {self.table.name}, "
                f"got {len(reducible_character)}"
            )
        
        decomposition = {}
        
        for irrep_name, irrep_char in self.irreps.items():
            # Inner product: Σ |G_c| * χ(c) * χ̄(c)
            inner_product = sum(
                self.class_sizes[i] * reducible_character[i] * 
                np.conj(irrep_char[i])
                for i in range(len(self.class_sizes))
            )
            
            multiplicity = inner_product / self.group_order
            multiplicity_int = round(multiplicity.real)
            
            if abs(multiplicity.real - multiplicity_int) > TOLERANCE:
                raise ValueError(
                    f"Multiplicity of {irrep_name} is {multiplicity.real}, "
                    f"not an integer. Check input character."
                )
            
            if multiplicity_int > 0:
                decomposition[irrep_name] = multiplicity_int
        
        return decomposition
    
    def tensor_product(self, char1: List, char2: List) -> Tuple[List, Dict[str, int]]:
        """
        Calculate tensor product of two characters
        
        χ₁ ⊗ χ₂ = χ₁(g) * χ₂(g)
        
        Args:
            char1, char2: Character vectors
            
        Returns:
            Tuple of (tensor character, decomposition)
        """
        tensor_char = [a * b for a, b in zip(char1, char2)]
        decomp = self.decompose(tensor_char)
        return tensor_char, decomp
    
    def direct_sum(self, char1: List, char2: List) -> Tuple[List, Dict[str, int]]:
        """
        Calculate direct sum of two representations
        
        χ₁ ⊕ χ₂ = χ₁(g) + χ₂(g)
        
        Args:
            char1, char2: Character vectors
            
        Returns:
            Tuple of (direct sum character, decomposition)
        """
        direct_sum_char = [a + b for a, b in zip(char1, char2)]
        decomp = self.decompose(direct_sum_char)
        return direct_sum_char, decomp
    
    def get_character_at_power(self, character: List, n: int) -> List:
        """
        Compute χ(g^n)
        
        Uses class_cycles to find the class containing g^n
        
        Args:
            character: Character vector
            n: Power
            
        Returns:
            Character vector of χ(g^n)
            
        Raises:
            ValueError: If class_cycles not defined
        """
        if self.class_cycles is None:
            raise ValueError(
                f"Class cycles not defined for {self.table.name}. "
                f"Cannot compute power characters."
            )
        
        power_char = []
        for i in range(len(self.class_sizes)):
            order, cycle = self.class_cycles[i]
            n_mod = n % order
            class_idx = cycle[n_mod]
            power_char.append(character[class_idx])
        
        return power_char
    
    def symmetric_product_general(self, character: List, n: int) -> List:
        """
        Calculate Sym^n(χ) using permutation group formula
        
        Sym^n(V) has character: (1/n!) Σ_{σ ∈ S_n} χ_{ind}(σ) Π_i χ(g^{c_i})
        
        where c_i is the cycle length in σ
        
        Args:
            character: Input character vector
            n: Order of symmetric product
            
        Returns:
            Character vector of Sym^n representation
        """
        if n == 0:
            return [1] * len(self.class_sizes)
        
        classes = self._conjugate_classes_sn(n)
        n_factorial = math.factorial(n)
        
        sym_char = [0] * len(self.class_sizes)
        
        for _, partition, class_size, _ in classes:
            for class_idx in range(len(self.class_sizes)):
                product = 1.0
                for cycle_len in partition:
                    power_char = self.get_character_at_power(character, cycle_len)
                    product *= power_char[class_idx]
                sym_char[class_idx] += class_size * product
        
        # Normalize and cleanup
        result = []
        for x in sym_char:
            normalized = self.normalize_number(x / n_factorial)
            result.append(normalized)
        
        return result
    
    def antisymmetric_product_general(self, character: List, n: int) -> List:
        """
        Calculate Alt^n(χ) using permutation group formula with sign
        
        Alt^n(V) has character: (1/n!) Σ_{σ ∈ S_n} sgn(σ) χ_{ind}(σ) Π_i χ(g^{c_i})
        
        Args:
            character: Input character vector
            n: Order of antisymmetric product
            
        Returns:
            Character vector of Alt^n representation
        """
        if n == 0:
            return [1] * len(self.class_sizes)
        
        classes = self._conjugate_classes_sn(n)
        n_factorial = math.factorial(n)
        
        antisym_char = [0] * len(self.class_sizes)
        
        for _, partition, class_size, sign in classes:
            for class_idx in range(len(self.class_sizes)):
                product = 1.0
                for cycle_len in partition:
                    power_char = self.get_character_at_power(character, cycle_len)
                    product *= power_char[class_idx]
                antisym_char[class_idx] += class_size * sign * product
        
        result = []
        for x in antisym_char:
            normalized = self.normalize_number(x / n_factorial)
            result.append(normalized)
        
        return result
    
    def symmetric_and_antisymmetric_products(self, character: List, n: int) -> Tuple:
        """Calculate both symmetric and antisymmetric products"""
        sym_char = self.symmetric_product_general(character, n)
        sym_decomp = self.decompose(sym_char)
        
        antisym_char = self.antisymmetric_product_general(character, n)
        antisym_decomp = self.decompose(antisym_char)
        
        return sym_char, sym_decomp, antisym_char, antisym_decomp
    
    # ==================== Verification Methods ====================
    
    def verify_table(self, verbose: bool = False) -> bool:
        """Verify character table consistency"""
        checks = {
            'class_count': self._check_class_count(),
            'dimension_sum': self._check_dimension_sum(),
            'dimensions_divide_order': self._check_dimensions_divide(),
            'row_orthogonality': self._check_row_orthogonality(),
            'column_orthogonality': self._check_column_orthogonality()
        }
        
        if verbose:
            print(f"\nVerifying {self.table.name}:")
            for check_name, passed in checks.items():
                status = "✓" if passed else "✗"
                print(f"  {status} {check_name}")
        
        return all(checks.values())
    
    def _check_class_count(self) -> bool:
        """Number of classes equals number of irreps"""
        return len(self.class_sizes) == len(self.irreps)
    
    def _check_dimension_sum(self) -> bool:
        """Σ dim² = |G|"""
        dim_sum = sum(
            int(round(char[0].real if isinstance(char[0], complex) else char[0])) ** 2
            for char in self.irreps.values()
        )
        return dim_sum == self.group_order
    
    def _check_dimensions_divide(self) -> bool:
        """Each dim divides |G|"""
        for char in self.irreps.values():
            dim = int(round(char[0].real if isinstance(char[0], complex) else char[0]))
            if self.group_order % dim != 0:
                return False
        return True
    
    def _check_row_orthogonality(self) -> bool:
        """⟨χ_i, χ_j⟩ = δ_ij * |G|"""
        irreps_list = list(self.irreps.items())
        for i, (name1, char1) in enumerate(irreps_list):
            for j, (name2, char2) in enumerate(irreps_list):
                inner_prod = sum(
                    self.class_sizes[k] * char1[k] * np.conj(char2[k])
                    for k in range(len(self.class_sizes))
                )
                inner_prod = round(inner_prod.real)
                expected = self.group_order if i == j else 0
                if abs(inner_prod - expected) > TOLERANCE:
                    return False
        return True
    
    def _check_column_orthogonality(self) -> bool:
        """Column orthogonality relations"""
        for i in range(len(self.class_sizes)):
            for j in range(len(self.class_sizes)):
                inner_prod = sum(
                    char[i] * np.conj(char[j])
                    for char in self.irreps.values()
                )
                inner_prod = round(inner_prod.real)
                expected = self.group_order // self.class_sizes[i] if i == j else 0
                if abs(inner_prod - expected) > TOLERANCE:
                    return False
        return True
    
    # ==================== Helper Methods ====================
    
    @staticmethod
    def _conjugate_classes_sn(n: int) -> List[Tuple]:
        """
        Find conjugacy classes of S_n (symmetric group)
        
        Each conjugacy class is characterized by a partition of n.
        Returns: List of (partition_description, partition, class_size, sign)
        """
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
            counts = Counter(partition)
            m = len(partition)
            sign = (-1) ** (n + m)
            
            denom = 1
            for cycle_len, count in counts.items():
                denom *= (cycle_len ** count) * math.factorial(count)
            
            size = math.factorial(n) // denom
            
            # Create a readable description of the partition
            # e.g., [2, 1] becomes "2+1" or [3] becomes "3"
            partition_desc = "+".join(str(p) for p in sorted(partition, reverse=True))
            
            classes.append((partition_desc, partition.copy(), size, sign))
        
        generate_partitions(n, n, [])
        classes.sort(key=lambda x: (len(x[1]), x[1]), reverse=True)
        return classes

    @staticmethod
    def format_decomposition(decomp: Dict[str, int]) -> str:
        """Format decomposition as string"""
        if not decomp:
            return "0 (empty)"
        
        parts = []
        for name, mult in sorted(decomp.items()):
            if mult == 1:
                parts.append(name)
            else:
                parts.append(f"{mult}{name}")
        
        return " ⊕ ".join(parts)
    
    def print_character_table(self):
        """Display character table"""
        print(f"\n{'=' * 100}")
        print(f"{self.table.name} Character Table")
        print(f"{'=' * 100}")
        
        header = f"{'Irrep':<15} " + " ".join(f"{cls:<12}" for cls in self.class_names)
        print(header)
        print("-" * 100)
        
        for irrep_name, character in self.irreps.items():
            formatted_char = []
            for val in character:
                if isinstance(val, complex):
                    formatted_char.append(f"{val:.4g}")
                else:
                    formatted_char.append(str(int(val) if isinstance(val, int) else val))
            
            row = f"{irrep_name:<15} " + " ".join(f"{val:<12}" for val in formatted_char)
            print(row)
        
        print("-" * 100)
        print(f"Order: {self.group_order}")
        print(f"Class sizes: {self.class_sizes}")
        
        if self.table.special_notes:
            print(f"Notes: {self.table.special_notes}")