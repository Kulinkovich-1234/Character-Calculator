"""
Expression parser for Character Calculator.

Grammar (confirmed by user):
    expression   → sum_expr (ARROW IDENT)?
    sum_expr     → product_expr ((PLUS) product_expr)*
    product_expr → factor ((TIMES) factor)*
    factor       → function
                 | BRACKET         → manual char | irrep ref | vec ref | stored ref
    function     → "Sym" "^" NUMBER "(" expression ")"
                 | "Alt" "^" NUMBER "(" expression ")"
                 | "Pow" "^" NUMBER "(" expression ")"
                 | "gPow" "^" NUMBER "(" expression ")"
                 | "Y" "(" arg ")"        arg = NUMBER | IDENT(orbital letter)
                 | "Poly" "(" NUMBER ")"
                 | "P" "(" NUMBER ")"

All character references MUST be enclosed in [...] brackets.
    [3, 0, -1, 1]         manual character (comma-separated numbers)
    [T1u]                 irrep reference
    [Vec]  or  [V]        vector representation
    [$name]               stored character

Outside brackets, `*` is a tensor-product operator (not part of identifiers).
The bracket rule eliminates all ambiguity between irrep names and operators.

Operator precedence (low → high):
    ⊕ +   (direct sum, left associative)
    ⊗ x * (tensor product, left associative)
    functions / atoms    (highest)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Union, Any
import re


# ========================================================================
# AST Node classes
# ========================================================================

class Node:
    """Base class for all AST nodes."""
    def __repr__(self):
        return f"{type(self).__name__}({', '.join(f'{k}={v!r}' for k, v in self.__dict__.items())})"


@dataclass
class Sum(Node):
    """Direct sum: a ⊕ b  or  a + b"""
    parts: List[tuple]  # list of (op: str, node: Node)


@dataclass
class Product(Node):
    """Tensor product: a ⊗ b  or  a x b  or  a * b"""
    parts: List[tuple]  # list of (op: str, node: Node)


@dataclass
class Sym(Node):
    """Symmetric power Sym^n(expr)"""
    n: int
    expr: Node


@dataclass
class Alt(Node):
    """Antisymmetric power Alt^n(expr)"""
    n: int
    expr: Node


@dataclass
class Pow(Node):
    """Tensor power χ⊗ⁿ  =  χ ⊗ χ ⊗ ... ⊗ χ (n times)"""
    n: int
    expr: Node


@dataclass
class GPow(Node):
    """Power character χ(gⁿ)"""
    n: int
    expr: Node


@dataclass
class Y(Node):
    """Spherical harmonic Y(l) where l is an integer or orbital letter"""
    l: int


@dataclass
class Poly(Node):
    """Polynomial representation Sym^n(Vec)"""
    n: int


# --- Bracket leaf nodes ---

@dataclass
class ManualChar(Node):
    """[v1, v2, v3, ...]  inline character values"""
    values: List[Union[float, complex]]


@dataclass
class IrrepRef(Node):
    """[T1u]  reference to an irreducible representation"""
    name: str


@dataclass
class VecRef(Node):
    """[Vec] or [V]  reference to the vector representation"""
    pass


@dataclass
class StoredRef(Node):
    """[$name]  reference to a stored character"""
    name: str


# --- Top-level wrapper ---

@dataclass
class Save:
    """expr -> name   or   expr -> (auto-name)"""
    expr: Node
    name: Optional[str] = None  # None = auto-generate name


# ========================================================================
# Tokenizer
# ========================================================================

@dataclass
class Token:
    kind: str    # token type
    value: Any   # token value (string or number)
    pos: int     # position in source string

    def __repr__(self):
        return f"Token({self.kind}, {self.value!r}, pos={self.pos})"


# Orbital letter → l mapping
ORBITAL_MAP = {
    's': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4, 'h': 5, 'i': 6,
    'j': 7, 'k': 8, 'l': 9, 'm': 10, 'n': 11, 'o': 12,
}


# Function names that take ( expression )
EXPR_FUNCTIONS = {'Sym', 'Alt', 'Pow', 'gPow'}

# Function names that take ( number / letter )
ARG_FUNCTIONS = {'Y', 'Poly', 'P'}

# All recognized function names
ALL_FUNCTIONS = EXPR_FUNCTIONS | ARG_FUNCTIONS


def tokenize(text: str) -> List[Token]:
    """
    Tokenize an expression string.

    Bracket content [...] is captured as a single BRACKET token with the
    raw content string. The parser decides whether it's manual numbers
    or an identifier reference.
    """
    tokens = []
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]

        # Skip whitespace
        if ch.isspace():
            i += 1
            continue

        # Arrow ->
        if ch == '-' and i + 1 < n and text[i + 1] == '>':
            tokens.append(Token('ARROW', '->', i))
            i += 2
            continue

        # Bracket [...] — capture all content until ]
        if ch == '[':
            start = i
            i += 1
            depth = 1
            content_parts = []
            while i < n and depth > 0:
                if text[i] == '[':
                    depth += 1
                elif text[i] == ']':
                    depth -= 1
                    if depth == 0:
                        break
                content_parts.append(text[i])
                i += 1
            if depth != 0:
                raise SyntaxError(f"Unclosed '[' at position {start}")
            content = ''.join(content_parts).strip()
            tokens.append(Token('BRACKET', content, start))
            i += 1  # skip ]
            continue

        # Delimiters
        if ch in '(),':
            kind_map = {'(': 'LPAREN', ')': 'RPAREN', ',': 'COMMA'}
            tokens.append(Token(kind_map[ch], ch, i))
            i += 1
            continue

        # Caret (for Sym^n, etc.)
        if ch == '^':
            tokens.append(Token('CARET', '^', i))
            i += 1
            continue

        # Unicode operators
        if ch in '⊗⊕':
            tokens.append(Token('OP', ch, i))
            i += 1
            continue

        # ASCII operators: + * x
        if ch in '+*x':
            tokens.append(Token('OP', ch, i))
            i += 1
            continue

        # Dollar sign (for stored references)
        if ch == '$':
            # If inside a bracket context, this is part of the content
            # and would be captured above. Here at the token level,
            # a bare $ is an error.
            raise SyntaxError(f"Unexpected '$' at position {i}: "
                              f"'$' is only valid inside [...]")

        # Negative/dash — at expression level, only valid as part of ->
        # which is already handled. Lone '-' is an error.
        if ch == '-':
            raise SyntaxError(f"Unexpected '-' at position {i}. "
                              f"Use '->' for save, not bare '-'.")

        # Number (integer or float)
        if ch.isdigit():
            start = i
            i += 1
            while i < n and (text[i].isdigit() or text[i] in '.eE'):
                if text[i] in 'eE':
                    i += 1
                    if i < n and text[i] in '+-':
                        i += 1
                    continue
                i += 1
            num_str = text[start:i]
            # Parse as int or float
            if '.' in num_str or 'e' in num_str.lower():
                tokens.append(Token('NUMBER', float(num_str), start))
            else:
                tokens.append(Token('NUMBER', int(num_str), start))
            continue

        # Identifier or keyword
        if ch.isalpha() or ch == '_':
            start = i
            i += 1
            # Only allow alphanumeric + underscore for keywords/idents
            # (irrep-specific chars like ' * are handled inside brackets)
            while i < n and (text[i].isalnum() or text[i] == '_'):
                i += 1
            ident = text[start:i]
            tokens.append(Token('IDENT', ident, start))
            continue

        # Any other character is an error
        raise SyntaxError(f"Unexpected character '{ch}' at position {i}")

    return tokens


# ========================================================================
# Recursive Descent Parser
# ========================================================================

class ParseError(Exception):
    def __init__(self, message: str, pos: Optional[int] = None):
        self.pos = pos
        msg = f"position {pos}: {message}" if pos is not None else message
        super().__init__(msg)


class Parser:
    """Recursive descent parser for character expressions."""

    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Optional[Token]:
        """Return current token without consuming."""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def consume(self, *expected_kinds: str) -> Token:
        """Consume and return the current token, checking its kind."""
        token = self.peek()
        if token is None:
            expected = '/'.join(expected_kinds)
            raise ParseError(f"Expected {expected}, got end of input",
                             self.pos)
        if expected_kinds and token.kind not in expected_kinds:
            raise ParseError(
                f"Expected {expected_kinds}, got {token.kind}('{token.value}')",
                token.pos)
        self.pos += 1
        return token

    def expect_end(self):
        """Ensure all tokens have been consumed."""
        if self.peek() is not None:
            tok = self.peek()
            raise ParseError(f"Unexpected token '{tok.value}' after end of expression",
                             tok.pos)

    # ----------------------------------------------------------------
    # Grammar rules
    # ----------------------------------------------------------------

    def parse(self) -> Union[Node, Save]:
        """
        Parse a complete expression:
            expression = sum_expr (ARROW IDENT)?
        """
        expr = self._parse_sum_expr()

        if self.peek() is not None and self.peek().kind == 'ARROW':
            self.consume('ARROW')
            if self.peek() is not None and self.peek().kind == 'IDENT':
                name_token = self.consume('IDENT')
                name = name_token.value
            else:
                name = None  # auto-name
            self.expect_end()
            return Save(expr, name)

        self.expect_end()
        return expr

    def _parse_sum_expr(self) -> Node:
        """sum_expr = product_expr ((PLUS) product_expr)*"""
        left = self._parse_product_expr()

        parts = []
        while self.peek() is not None and self.peek().kind == 'OP' \
                and self.peek().value in ('+', '⊕'):
            op = self.consume('OP').value
            right = self._parse_product_expr()
            parts.append((op, right))

        if not parts:
            return left
        all_parts = [(None, left)] + parts
        node = Sum(all_parts)
        return node

    def _parse_product_expr(self) -> Node:
        """product_expr = factor ((TIMES) factor)*"""
        left = self._parse_factor()

        parts = []
        while self.peek() is not None and self.peek().kind == 'OP' \
                and self.peek().value in ('⊗', 'x', '*'):
            op = self.consume('OP').value
            right = self._parse_factor()
            parts.append((op, right))

        if not parts:
            return left
        all_parts = [(None, left)] + parts
        node = Product(all_parts)
        return node

    def _parse_factor(self) -> Node:
        """factor = function | BRACKET | LPAREN expression RPAREN"""
        token = self.peek()
        if token is None:
            raise ParseError("Unexpected end of input, expected expression",
                             self.pos)

        # Parenthesized sub-expression
        if token.kind == 'LPAREN':
            self.consume('LPAREN')
            expr = self._parse_sum_expr()
            self.consume('RPAREN')
            return expr

        # Bracket content (manual char, irrep ref, vec ref, stored ref)
        if token.kind == 'BRACKET':
            return self._parse_bracket(self.consume('BRACKET'))

        # Function call
        if token.kind == 'IDENT':
            return self._parse_function(self.consume('IDENT'))

        raise ParseError(f"Unexpected token '{token.value}'",
                         token.pos)

    # ----------------------------------------------------------------
    # Bracket content
    # ----------------------------------------------------------------

    def _parse_bracket(self, token: Token) -> Node:
        """
        Parse the content of [...] into a leaf AST node.
        One of: ManualChar, IrrepRef, VecRef, StoredRef
        """
        content = token.value

        # Stored reference: [$name]
        if content.startswith('$'):
            name = content[1:].strip()
            if not name:
                raise ParseError("Empty stored character reference '$'",
                                 token.pos)
            return StoredRef(name)

        # Vector representation: [Vec] or [V]
        if content.lower() in ('vec', 'v'):
            return VecRef()

        # Manual character: number list like [3, 0, -1, 1]
        # Detection heuristic: if the content (stripped) consists only of
        # characters valid in number expressions, try to parse as numbers.
        stripped = content.strip()
        if _looks_like_number_list(stripped):
            try:
                values = _parse_number_list(stripped)
                return ManualChar(values)
            except ValueError:
                pass  # fall through to IrrepRef

        # Irrep reference: [T1u], [E1'*], etc.
        return IrrepRef(stripped)

    # ----------------------------------------------------------------
    # Functions
    # ----------------------------------------------------------------

    def _parse_function(self, token: Token) -> Node:
        """Parse a function call like Sym^n(expr), Y(arg), etc."""
        name = token.value

        if name not in ALL_FUNCTIONS:
            raise ParseError(
                f"Unknown function '{name}' at expression level. "
                f"Expected one of: {', '.join(sorted(ALL_FUNCTIONS))}, "
                f"or a bracket [...] expression.",
                token.pos)

        # --- Expression-taking functions: Sym^n(expr), Alt^n(expr), etc.
        if name in EXPR_FUNCTIONS:
            self.consume('CARET')
            n_token = self.consume('NUMBER')
            n = int(n_token.value) if isinstance(n_token.value, int) \
                else int(n_token.value)
            self.consume('LPAREN')
            expr = self._parse_sum_expr()
            self.consume('RPAREN')

            if name == 'Sym':
                return Sym(n, expr)
            elif name == 'Alt':
                return Alt(n, expr)
            elif name == 'Pow':
                return Pow(n, expr)
            elif name == 'gPow':
                return GPow(n, expr)

        # --- Argument-taking functions: Y(l), Poly(n), P(n)
        elif name in ARG_FUNCTIONS:
            self.consume('LPAREN')

            arg_token = self.peek()
            if arg_token is None:
                raise ParseError(f"Expected argument for {name}()", token.pos)

            if name == 'Y':
                # Y( number )  or  Y( identifier / orbital letter )
                if arg_token.kind == 'NUMBER':
                    self.consume('NUMBER')
                    l = int(arg_token.value)
                elif arg_token.kind == 'IDENT':
                    self.consume('IDENT')
                    letter = arg_token.value.lower()
                    if letter in ORBITAL_MAP:
                        l = ORBITAL_MAP[letter]
                    else:
                        raise ParseError(
                            f"Unknown orbital letter '{arg_token.value}'. "
                            f"Use a number (0,1,2,...) or letter (s,p,d,f,...).",
                            arg_token.pos)
                else:
                    raise ParseError(
                        f"Expected number or orbital letter for Y(), "
                        f"got {arg_token.kind}('{arg_token.value}')",
                        arg_token.pos)
                self.consume('RPAREN')
                return Y(l)

            elif name in ('Poly', 'P'):
                if arg_token.kind == 'NUMBER':
                    self.consume('NUMBER')
                    n = int(arg_token.value)
                else:
                    raise ParseError(
                        f"Expected number for {name}(), "
                        f"got {arg_token.kind}",
                        arg_token.pos)
                self.consume('RPAREN')
                return Poly(n)

        # Should not reach here
        raise ParseError(f"Internal error: unhandled function '{name}'",
                         token.pos)


# ========================================================================
# Helper functions for bracket content parsing
# ========================================================================

# Valid characters in a number list — allows: digits, ., -, +, e, E, j, i, comma, whitespace
_NUMBER_LIST_CHARS = set('0123456789.+-eEjJiI, \t')


def _looks_like_number_list(s: str) -> bool:
    """Check if a string looks like a comma-separated list of numbers."""
    if not s:
        return False
    # Must only contain number-valid characters
    for ch in s:
        if ch not in _NUMBER_LIST_CHARS:
            return False
    # Must contain at least one digit
    if not any(ch.isdigit() for ch in s):
        return False
    return True


def _parse_number_list(s: str) -> List[Union[float, complex]]:
    """Parse a comma-separated string of numbers into a list."""
    parts = [p.strip() for p in s.split(',')]
    values = []
    for p in parts:
        if not p:
            continue
        # Normalize 'i' to 'j' for Python complex literal
        normalized = p.replace('i', 'j')
        if 'j' in normalized.lower():
            values.append(complex(normalized))
        else:
            values.append(float(normalized))
    return values


# ========================================================================
# Top-level convenience
# ========================================================================

def parse_expression(expr_str: str) -> Union[Node, Save]:
    """
    Parse an expression string into an AST node.

    Args:
        expr_str: The expression string (e.g. "Sym^2([T1u]) ⊗ [Eg]")

    Returns:
        A Node (or Save wrapper) for the expression.

    Raises:
        SyntaxError, ParseError: On invalid syntax.
    """
    tokens = tokenize(expr_str)
    parser = Parser(tokens)
    return parser.parse()


def expression_type(node: Union[Node, Save]) -> str:
    """Return a human-readable type name for an AST node."""
    return type(node).__name__
