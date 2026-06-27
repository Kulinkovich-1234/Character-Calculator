"""
Expression evaluator for Character Calculator.

Walks the AST produced by expression_parser.py and computes character
vectors by calling methods on CharacterCalculator and CharacterStorage.
"""

from __future__ import annotations
from typing import List, Union, Optional, Dict
import re

from character_calculator import CharacterCalculator
from character_storage import CharacterStorage
from expression_parser import (
    Node, Sum, Product, Sym, Alt, Pow, GPow, Y, Poly,
    ManualChar, IrrepRef, VecRef, StoredRef, Save,
    ORBITAL_MAP,
)


class EvalError(Exception):
    """Error during expression evaluation."""
    pass


class ExpressionEvaluator:
    """
    Recursive AST evaluator.

    Each `eval_*` method returns a character vector (list of numbers).
    Decomposition into irreps is done separately by the caller.
    """

    def __init__(self, calculator: CharacterCalculator,
                 storage: Optional[CharacterStorage] = None):
        self.calc = calculator
        self.storage = storage
        self.class_count = len(calculator.class_sizes)

    # ================================================================
    # Public entry point
    # ================================================================

    def eval(self, node: Node) -> List[Union[float, complex]]:
        """
        Evaluate an AST node and return the character vector.
        """
        # Dispatch by type
        if isinstance(node, Sum):
            return self._eval_sum(node)
        elif isinstance(node, Product):
            return self._eval_product(node)
        elif isinstance(node, Sym):
            return self._eval_sym(node)
        elif isinstance(node, Alt):
            return self._eval_alt(node)
        elif isinstance(node, Pow):
            return self._eval_pow(node)
        elif isinstance(node, GPow):
            return self._eval_gpow(node)
        elif isinstance(node, Y):
            return self._eval_y(node)
        elif isinstance(node, Poly):
            return self._eval_poly(node)
        elif isinstance(node, ManualChar):
            return self._eval_manual(node)
        elif isinstance(node, IrrepRef):
            return self._eval_irrep(node)
        elif isinstance(node, VecRef):
            return self._eval_vec(node)
        elif isinstance(node, StoredRef):
            return self._eval_stored(node)
        else:
            raise EvalError(f"Unknown node type: {type(node).__name__}")

    # ================================================================
    # Internal evaluation methods
    # ================================================================

    def _eval_sum(self, node: Sum) -> List[Union[float, complex]]:
        """Direct sum: pointwise addition of character vectors."""
        result = None
        for _op, subnode in node.parts:
            vec = self.eval(subnode)
            if result is None:
                result = list(vec)
            else:
                result = [a + b for a, b in zip(result, vec)]
        return result

    def _eval_product(self, node: Product) -> List[Union[float, complex]]:
        """Tensor product: pointwise multiplication of character vectors."""
        result = None
        for _op, subnode in node.parts:
            vec = self.eval(subnode)
            if result is None:
                result = list(vec)
            else:
                result = [a * b for a, b in zip(result, vec)]
        return result

    def _eval_sym(self, node: Sym) -> List[Union[float, complex]]:
        """Symmetric power Sym^n."""
        vec = self.eval(node.expr)
        return self.calc.symmetric_product_general(vec, node.n)

    def _eval_alt(self, node: Alt) -> List[Union[float, complex]]:
        """Antisymmetric power Alt^n."""
        vec = self.eval(node.expr)
        return self.calc.antisymmetric_product_general(vec, node.n)

    def _eval_pow(self, node: Pow) -> List[Union[float, complex]]:
        """Tensor power χ⊗ⁿ = pointwise power of the character vector."""
        vec = self.eval(node.expr)
        if node.n == 0:
            # Trivial representation
            return [1] * self.class_count
        # Python's pow with complex handles both real and complex vectors
        result = [pow(v, node.n) for v in vec]
        return result
        # Note: Pow^n(expr) = expr ⊗ expr ⊗ ... ⊗ expr = χ(g)^n
        # This is tensor power, not symmetric/antisymmetric power.

    def _eval_gpow(self, node: GPow) -> List[Union[float, complex]]:
        """Power character χ(g^n)."""
        vec = self.eval(node.expr)
        return self.calc.get_character_at_power(vec, node.n)

    def _eval_y(self, node: Y) -> List[Union[float, complex]]:
        """Spherical harmonic Y(l)."""
        if node.l < 0:
            raise EvalError(f"Angular quantum number cannot be negative: {node.l}")
        if self.calc.vector_char is None:
            raise EvalError(
                f"Vector representation not defined for {self.calc.table.name}. "
                f"Cannot compute spherical harmonics.")
        return self.calc.harmonic_character(node.l)

    def _eval_poly(self, node: Poly) -> List[Union[float, complex]]:
        """Polynomial representation Sym^n(Vec)."""
        if node.n < 0:
            raise EvalError(f"Polynomial degree cannot be negative: {node.n}")
        if self.calc.vector_char is None:
            raise EvalError(
                f"Vector representation not defined for {self.calc.table.name}. "
                f"Cannot compute polynomial character.")
        return self.calc.polynomial_character(node.n)

    def _eval_manual(self, node: ManualChar) -> List[Union[float, complex]]:
        """Manual character values."""
        if len(node.values) != self.class_count:
            raise EvalError(
                f"Expected {self.class_count} character values for "
                f"{self.calc.table.name}, got {len(node.values)}")
        return node.values

    def _eval_irrep(self, node: IrrepRef) -> List[Union[float, complex]]:
        """Reference to an irreducible representation."""
        name = node.name
        # Exact match first
        if name in self.calc.irreps:
            return list(self.calc.irreps[name])
        # Case-insensitive fallback
        for key in self.calc.irreps:
            if key.lower() == name.lower():
                return list(self.calc.irreps[key])
        raise EvalError(
            f"Irrep '{name}' not found in {self.calc.table.name}. "
            f"Available: {', '.join(self.calc.irreps.keys())}")

    def _eval_vec(self, node: VecRef) -> List[Union[float, complex]]:
        """Vector representation."""
        if self.calc.vector_char is None:
            raise EvalError(
                f"Vector representation not defined for {self.calc.table.name}.")
        return list(self.calc.vector_char)

    def _eval_stored(self, node: StoredRef) -> List[Union[float, complex]]:
        """Stored character reference."""
        if self.storage is None:
            raise EvalError("Character storage not available.")
        result = self.storage.get_character(self.calc.table.name, node.name)
        if result is None:
            raise EvalError(
                f"Stored character '{node.name}' not found for "
                f"{self.calc.table.name}.")
        char, _desc = result
        return list(char)


# ================================================================
# Utility functions for CLI display
# ================================================================

def format_character_vector(vec: List[Union[float, complex]],
                            tolerance: float = 1e-10) -> str:
    """Format a character vector for display."""
    parts = []
    for v in vec:
        if isinstance(v, complex):
            if abs(v.imag) < tolerance:
                v = v.real
            else:
                parts.append(f"{v.real:.6g}{v.imag:+.6g}j")
                continue
        # Real or real-only
        if isinstance(v, float) or isinstance(v, complex):
            if abs(v - round(v)) < tolerance:
                parts.append(str(int(round(v))))
            else:
                parts.append(f"{v:.6g}")
        else:
            parts.append(str(v))
    return "[" + ", ".join(parts) + "]"


def make_auto_name(expr_str: str) -> str:
    """Generate a storage name from an expression string."""
    name = expr_str.strip()
    # Remove save arrow if present
    name = re.sub(r'\s*->\s*$', '', name)
    name = name.replace('⊗', 'x')
    name = name.replace('⊕', '+')
    # Remove characters that might be problematic
    name = re.sub(r'[\[\]()^,]+', '', name)
    name = re.sub(r'\s+', '_', name)
    name = name.strip('_')
    name = name.strip('.')
    if not name:
        return "unnamed_result"
    if len(name) > 50:
        name = name[:50]
    return name
