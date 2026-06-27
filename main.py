"""
Main entry point - integrates all modules

CLI mode:
    python main.py                           Interactive mode
    python main.py list                      List available groups
    python main.py -g O_h "Sym^2([T1u])"     Evaluate expression
    python main.py -g O_h IR                 IR active irreps
    python main.py -g O_h Raman              Raman active irreps
    python main.py -g O_h table              Show character table
    python main.py verify --all              Verify all tables
    python main.py -g O_h verify             Verify one table
    python main.py storage --group O_h list  List stored characters
"""

import sys
import argparse
import json
from typing import Optional

from character_table_database import CharacterTableDatabase
from character_calculator import CharacterCalculator
from calculator_ui import CalculatorUI
from character_storage import CharacterStorage
from constants import __version__, __author__, __license__, CATEGORY_ORDER
from expression_parser import parse_expression, Save, ParseError
from expression_evaluator import ExpressionEvaluator, EvalError, \
    format_character_vector, make_auto_name


# ========================================================================
# Interactive mode (unchanged from original)
# ========================================================================

def display_welcome():
    """Display welcome message"""
    print("\n" + "=" * 80)
    print("Character Table Decomposer")
    print("=" * 80)
    print(f"Version: {__version__}")
    print(f"Author: {__author__}")
    print(f"License: {__license__}")
    print("\nFeatures:")
    print("  ✓ 40+ point groups")
    print("  ✓ Character decomposition")
    print("  ✓ Tensor products & direct sums")
    print("  ✓ Symmetric & antisymmetric powers")
    print("  ✓ Spherical harmonics / Atomic orbitals")
    print("  ✓ Polynomials (Sym^n)")
    print("  ✓ IR and Raman active modes")
    print("  ✓ Power characters χ(g^n)")
    print("  ✓ Character storage to JSON")
    print("  ✓ Table verification")
    print("  ✓ Expression CLI mode")
    print("=" * 80)


def display_groups(db: CharacterTableDatabase) -> dict:
    """Display available groups organized by category"""
    print("\nAvailable Point Groups:")
    print("=" * 80)

    groups_by_category = {}
    for group_name in db.list_groups():
        table = db.get_table(group_name)
        cat = table.category
        if cat not in groups_by_category:
            groups_by_category[cat] = []
        groups_by_category[cat].append(group_name)

    idx = 1
    display_map = {}

    for category in CATEGORY_ORDER:
        if category in groups_by_category:
            print(f"\n{category}:")
            for group_name in sorted(groups_by_category[category]):
                print(f"  {idx:2d}. {group_name}")
                display_map[idx] = group_name
                idx += 1

    return display_map


def verify_all_tables(db: CharacterTableDatabase):
    """Verify all character tables with enhanced checks"""
    print("\nVerifying all character tables...")
    print("=" * 80)

    passed = 0
    failed = []

    for group_name in sorted(db.list_groups()):
        try:
            table = db.get_table(group_name)
            calculator = CharacterCalculator(table)

            if calculator.verify_table(verbose=False):
                print(f"✓ {group_name}")
                passed += 1
            else:
                print(f"✗ {group_name}")
                failed.append(group_name)
        except Exception as e:
            print(f"✗ {group_name}: {e}")
            failed.append(group_name)

    print("=" * 80)
    print(f"\nResults: {passed} passed, {len(failed)} failed")
    if failed:
        print(f"Failed groups: {', '.join(failed)}")


def run_group_session(db: CharacterTableDatabase, storage: CharacterStorage,
                      group_name: str):
    """Run interactive session for a group"""
    try:
        table = db.get_table(group_name)
        calculator = CharacterCalculator(table)
        ui = CalculatorUI(calculator, storage)
        ui.run_interactive_session()
    except Exception as e:
        print(f"✗ Error: {e}")


def run_interactive_mode(db: CharacterTableDatabase,
                         storage: CharacterStorage):
    """Original interactive menu loop."""
    display_welcome()

    print("\nLoading character tables...")
    print(f"✓ Loaded {len(db.list_groups())} point groups")

    while True:
        try:
            display_map = display_groups(db)

            print(f"\n  V. Verify all tables")
            print(f"  0. Exit")

            choice = input(
                f"\nSelect group (1-{len(display_map)}, 0 to exit): "
            ).strip()

            if choice == "0":
                print("\nThank you for using Character Table Calculator!")
                break

            if choice.lower() == "v":
                verify_all_tables(db)
                continue

            try:
                idx = int(choice)
                if idx in display_map:
                    group_name = display_map[idx]
                    run_group_session(db, storage, group_name)
                else:
                    print(f"✗ Invalid selection")
            except ValueError:
                print(f"✗ Please enter a number")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"✗ Error: {e}")


# ========================================================================
# Output formatting helpers
# ========================================================================

def format_chars_with_classes(class_names: list,
                               char_vec: list,
                               tolerance: float = 1e-10) -> str:
    """Format a character vector aligned under class name columns."""
    # Build display values
    display_vals = []
    for v in char_vec:
        if isinstance(v, complex):
            if abs(v.imag) < tolerance:
                v = v.real
            else:
                display_vals.append(f"{v.real:.4g}{v.imag:+.4g}i")
                continue
        if abs(v - round(v)) < tolerance:
            display_vals.append(str(int(round(v))))
        elif isinstance(v, float):
            display_vals.append(f"{v:.4g}")
        else:
            display_vals.append(str(v))

    # Determine column widths (minimum 6 chars per column)
    col_widths = []
    for name, val_str in zip(class_names, display_vals):
        w = max(len(name), len(val_str), 6) + 2
        col_widths.append(w)

    # Build header: class names
    header = ""
    for name, w in zip(class_names, col_widths):
        header += f"{name:<{w}}"

    # Build value line (same indent as header)
    val_line = ""
    for val_str, w in zip(display_vals, col_widths):
        val_line += f"{val_str:<{w}}"

    return header + "\n" + val_line

    return header + "\n" + val_line


def print_chars_with_classes(class_names: list, char_vec: list,
                              indent: int = 2):
    """Print class names header and values aligned, with given indent."""
    fmt = format_chars_with_classes(class_names, char_vec)
    pad = " " * indent
    for line in fmt.split('\n'):
        print(f"{pad}{line}")


def _col_widths(class_names: list, display_vals: list) -> list:
    """Compute column widths for aligned display."""
    widths = []
    for name, val_str in zip(class_names, display_vals):
        w = max(len(name), len(val_str), 6) + 2
        widths.append(w)
    return widths


def _format_values(values: list, col_widths: list) -> str:
    """Format a list of values with given column widths."""
    return "".join(f"{v:<{w}}" for v, w in zip(values, col_widths))


def _vec_display(vec: list, tolerance: float = 1e-10) -> list:
    """Convert a character vector to clean display strings."""
    result = []
    for v in vec:
        if isinstance(v, complex):
            if abs(v.imag) < tolerance:
                v = v.real
            else:
                result.append(f"{v.real:.4g}{v.imag:+.4g}i")
                continue
        if abs(v - round(v)) < tolerance:
            result.append(str(int(round(v))))
        elif isinstance(v, float):
            result.append(f"{v:.4g}")
        else:
            result.append(str(v))
    return result

def build_cli_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description='Character Table Calculator',
        add_help=False,
    )
    parser.add_argument('--group', '-g', type=str,
                        help='Point group name (required for expressions)')
    parser.add_argument('--json', action='store_true',
                        help='Output results as JSON')
    parser.add_argument('--help', '-h', action='store_true',
                        help='Show this help message')
    return parser


CLI_HELP = """\
Character Table Calculator — CLI mode

Commands:
  list                              List available point groups
  verify [--all] [group]            Verify character table(s)
  storage --group GROUP [action]    Manage stored characters

  --group GROUP table               Show character table
  --group GROUP IR                  Show IR active irreps
  --group GROUP Raman               Show Raman active irreps
  --group GROUP verify              Verify the group's table
  --group GROUP "expression"        Evaluate an expression

Expressions:
  [T1u] x [Eg]                      Tensor product
  Sym^2([T1u])                      Symmetric power
  Alt^2([T1u])                      Antisymmetric power
  Pow^3([T1u])                      Tensor power (chi^n)
  gPow^2([T1u])                     Power character chi(g^n)
  Y(3)                              Spherical harmonic (l=3 / f-orbital)
  Poly(3)                           Polynomial Sym^3(Vec)
  [3, 0, -1, 1]                     Manual character
  [Vec]                             Vector representation
  [$name]                           Stored character

  expr -> name                      Save result as 'name'
  expr ->                           Save with auto-generated name

Example:
  python main.py -g O_h "Sym^2([T1u]) x [Eg]"
  python main.py -g O_h IR
  python main.py verify --all
"""


def list_groups_cli(db: CharacterTableDatabase):
    """Print available groups (CLI version)."""
    print("Available Point Groups:")
    for category in CATEGORY_ORDER:
        groups = db.get_groups_by_category(category)
        if groups:
            print(f"\n  {category}:")
            for name in sorted(groups):
                print(f"    {name}")


def verify_one(db: CharacterTableDatabase, group_name: str):
    """Verify a single group and print results."""
    try:
        table = db.get_table(group_name)
        calculator = CharacterCalculator(table)
        result = calculator.verify_table(verbose=True)
        if result:
            print(f"\n✓ {group_name}: All checks passed")
        else:
            print(f"\n✗ {group_name}: Some checks failed")
    except Exception as e:
        print(f"✗ {group_name}: {e}")


def handle_group_cmd(db: CharacterTableDatabase,
                     storage: CharacterStorage,
                     group_name: str, cmd: str,
                     json_output: bool = False):
    """Handle group-specific commands: IR, Raman, table, verify."""
    try:
        table = db.get_table(group_name)
        calculator = CharacterCalculator(table)
    except ValueError as e:
        print(f"Error: {e}")
        return

    if cmd == 'ir':
        if calculator.vector_char is None:
            print(f"Vector representation not defined for {group_name}")
            return
        irreps = calculator.get_ir_active_irreps()
        decomp = calculator.decompose(calculator.vector_char)
        if json_output:
            print(json.dumps({
                'group': group_name,
                'type': 'ir_active',
                'decomposition': decomp,
            }, ensure_ascii=False))
        else:
            print(f"{group_name} IR active irreps:")
            print(f"  {calculator.format_decomposition(decomp)}")

    elif cmd == 'raman':
        if calculator.vector_char is None:
            print(f"Vector representation not defined for {group_name}")
            return
        raman_irreps = calculator.get_raman_active_irreps()
        sym2_char = calculator.symmetric_product_general(
            calculator.vector_char, 2)
        decomp = calculator.decompose(sym2_char)
        if json_output:
            print(json.dumps({
                'group': group_name,
                'type': 'raman_active',
                'decomposition': decomp,
            }, ensure_ascii=False))
        else:
            print(f"{group_name} Raman active irreps:")
            print(f"  {calculator.format_decomposition(decomp)}")

    elif cmd == 'table':
        calculator.print_character_table()

    elif cmd == 'verify':
        calculator.verify_table(verbose=True)


def handle_storage_cli(db: CharacterTableDatabase,
                       storage: CharacterStorage,
                       group_name: Optional[str],
                       rest_args: list):
    """Handle 'storage' subcommand."""
    if not group_name:
        print("Error: storage requires --group GROUP")
        print("Usage: storage --group GROUP [list|export|delete] [name]")
        return

    if group_name not in db.list_groups():
        print(f"Error: Unknown group '{group_name}'")
        return

    # action is the first non-flag token after 'storage', if any
    action_tokens = [a for a in rest_args if not a.startswith('-')]
    action = action_tokens[1].lower() if len(action_tokens) > 1 else 'list'

    if action == 'list':
        chars = storage.list_stored_characters(group_name)
        if chars:
            print(f"Stored characters for {group_name}:")
            for c in chars:
                print(f"  - {c}")
        else:
            print(f"No stored characters for {group_name}")

    elif action == 'export':
        name = action_tokens[2] if len(action_tokens) > 2 else None
        if not name:
            print("Error: export requires a character name")
            return
        storage.export_to_csv(group_name, name, f"{name}.csv")

    elif action == 'delete':
        name = action_tokens[2] if len(action_tokens) > 2 else None
        if not name:
            print("Error: delete requires a character name")
            return
        storage.delete_character(group_name, name)

    else:
        print(f"Unknown storage action '{action}'. Use list, export, delete.")


def evaluate_expression(db: CharacterTableDatabase,
                        storage: CharacterStorage,
                        group_name: str,
                        expr_str: str,
                        json_output: bool = False):
    """Parse, evaluate, and display an expression result."""
    try:
        table = db.get_table(group_name)
    except ValueError as e:
        print(f"Error: {e}")
        return

    calculator = CharacterCalculator(table)
    evaluator = ExpressionEvaluator(calculator, storage)

    # 1) Parse
    try:
        ast = parse_expression(expr_str)
    except (SyntaxError, ParseError) as e:
        print(f"Error parsing expression: {e}")
        return

    # 2) Determine if saving is requested
    save_name = None
    if isinstance(ast, Save):
        eval_node = ast.expr
        if ast.name is not None:
            save_name = ast.name
        else:
            save_name = make_auto_name(expr_str)
    else:
        eval_node = ast

    # 3) Evaluate
    try:
        char_vec = evaluator.eval(eval_node)
    except (EvalError, ValueError) as e:
        print(f"Error evaluating expression: {e}")
        return

    # 4) Decompose
    try:
        decomp = calculator.decompose(char_vec)
    except ValueError as e:
        # May fail for non-integer multiplicities (pure manual input, etc.)
        decomp = None

    # 5) Output
    if json_output:
        result = {
            'expression': expr_str,
            'group': group_name,
            'character': [complex(c).real if abs(complex(c).imag) < 1e-10
                          else str(c) for c in char_vec],
        }
        if decomp is not None:
            result['decomposition'] = decomp
        if save_name is not None:
            result['saved_as'] = save_name
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(f"{expr_str}")
        print_chars_with_classes(calculator.class_names, char_vec)
        if decomp is not None:
            decomp_str = calculator.format_decomposition(decomp)
            print(f"  Decomposition: {decomp_str}")

    # 6) Save if requested
    if save_name is not None:
        description = f"Expression: {expr_str}"
        storage.store_character(group_name, save_name, char_vec, description)
        if not json_output:
            print(f"  ✓ Saved as '{save_name}'")


# ========================================================================
# Vibration analysis
# ========================================================================

def handle_vibration(db: CharacterTableDatabase,
                     storage: CharacterStorage,
                     group_name: str,
                     fixed_atoms_args: list):
    """
    Perform molecular vibration mode analysis.

    Computes Γ_total → Γ_trans → Γ_rot → Γ_vib, then shows IR/Raman activity.
    """
    try:
        table = db.get_table(group_name)
    except ValueError as e:
        print(f"Error: {e}")
        return

    calculator = CharacterCalculator(table)
    class_names = calculator.class_names
    n_classes = len(class_names)
    vec = calculator.vector_char

    if vec is None:
        print(f"Error: Vector representation not defined for {group_name}.")
        return

    # ---- Input: fixed atom counts ----
    if fixed_atoms_args:
        # From command line
        try:
            fixed = [int(x) for x in fixed_atoms_args]
        except ValueError:
            print("Error: Fixed atom counts must be integers.")
            return
    else:
        # Interactive prompt
        print(f"\n  Classes: {', '.join(class_names)}")
        prompt = f"  Enter fixed atoms per class ({n_classes} numbers, space-separated): "
        while True:
            raw = input(prompt).strip()
            parts = raw.split()
            if len(parts) != n_classes:
                print(f"  Expected {n_classes} numbers, got {len(parts)}. Try again.")
                continue
            try:
                fixed = [int(x) for x in parts]
                break
            except ValueError:
                print("  All values must be integers. Try again.")
                continue

    if len(fixed) != n_classes:
        print(f"Error: Expected {n_classes} fixed atom counts, got {len(fixed)}.")
        return

    # ---- Compute characters ----
    total_char = [fixed[i] * vec[i] for i in range(n_classes)]
    trans_char = list(vec)
    rot_char = calculator.antisymmetric_product_general(vec, 2)
    vib_char = [total_char[i] - trans_char[i] - rot_char[i]
                for i in range(n_classes)]

    # ---- Decompose ----
    try:
        total_decomp = calculator.decompose(total_char)
    except ValueError as e:
        total_decomp = {}
    trans_decomp = calculator.decompose(trans_char)
    rot_decomp = calculator.decompose(rot_char)

    try:
        vib_decomp = calculator.decompose(vib_char)
    except ValueError as e:
        vib_decomp = {}

    # ---- IR / Raman ----
    try:
        ir_irreps = calculator.get_ir_active_irreps()
        ir_list = calculator.decompose(vec)
    except (ValueError, EvalError):
        ir_irreps = []
        ir_list = {}

    try:
        raman_irreps = calculator.get_raman_active_irreps()
        raman_sym2 = calculator.symmetric_product_general(vec, 2)
        raman_list = calculator.decompose(raman_sym2)
    except (ValueError, EvalError):
        raman_irreps = []
        raman_list = {}

    # ---- Count multiplicities in Γ_vib ----
    ir_in_vib = {ir: vib_decomp.get(ir, 0) for ir in ir_list}
    raman_in_vib = {ir: vib_decomp.get(ir, 0) for ir in raman_list}

    total_ir_modes = sum(ir_in_vib.values()) if ir_in_vib else 0
    total_raman_modes = sum(raman_in_vib.values()) if raman_in_vib else 0

    # ---- Display ----
    print()
    print(f"{'=' * 60}")
    print(f"  {group_name} Vibration Mode Analysis")
    print(f"{'=' * 60}")
    print()

    # Compute dynamic column widths from class names and values
    display_vals = _vec_display(total_char)
    cw = _col_widths(class_names, display_vals)

    # Class names reference line
    print(f"  {'Classes:':<20}" + _format_values(class_names, cw))
    print(f"  {'Fixed atoms:':<20}" + _format_values(fixed, cw))
    print()

    # Γ_total
    total_str = calculator.format_decomposition(total_decomp)
    td = _vec_display(total_char)
    print(f"  Γ_total = N_fixed × χ_vec")
    print_chars_with_classes(class_names, total_char, indent=4)
    print(f"    → {total_str}")
    print()

    # Γ_trans
    trans_str = calculator.format_decomposition(trans_decomp)
    print(f"  Γ_trans = Vec")
    print_chars_with_classes(class_names, trans_char, indent=4)
    print(f"    → {trans_str}")
    print()

    # Γ_rot
    rot_str = calculator.format_decomposition(rot_decomp)
    print(f"  Γ_rot = Alt²(Vec)")
    print_chars_with_classes(class_names, rot_char, indent=4)
    print(f"    → {rot_str}")
    print()

    # Γ_vib
    vib_str = calculator.format_decomposition(vib_decomp)
    print(f"  Γ_vib = Γ_total − Γ_trans − Γ_rot")
    print_chars_with_classes(class_names, vib_char, indent=4)
    print(f"    → {vib_str}")
    print()

    # IR activity
    print(f"  {'─' * 50}")
    ir_active_str = calculator.format_decomposition(ir_list)
    print(f"  IR active (Vec decomposition):")
    print(f"    {ir_active_str}")
    if ir_in_vib:
        details = ", ".join(f"{ir} ×{mult}"
                           for ir, mult in sorted(ir_in_vib.items()) if mult > 0)
        print(f"    → in Γ_vib: {details}")
        print(f"    → {total_ir_modes} IR-active vibration mode(s)")
    print()

    # Raman activity
    raman_active_str = calculator.format_decomposition(raman_list)
    print(f"  Raman active (Sym²(Vec) decomposition):")
    print(f"    {raman_active_str}")
    if raman_in_vib:
        details = ", ".join(f"{ir} ×{mult}"
                           for ir, mult in sorted(raman_in_vib.items()) if mult > 0)
        print(f"    → in Γ_vib: {details}")
        print(f"    → {total_raman_modes} Raman-active vibration mode(s)")
    print(f"{'=' * 60}")
# ========================================================================

def main():
    """Main entry point — interactive or CLI mode."""
    parser = build_cli_parser()
    args, rest = parser.parse_known_args()

    # Show help if requested
    if args.help:
        print(CLI_HELP)
        return

    # Determine mode: CLI mode if there are arguments
    is_cli_mode = bool(rest) or bool(args.group)

    # Initialize database and storage
    try:
        db = CharacterTableDatabase()
        storage = CharacterStorage(quiet=is_cli_mode)
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        sys.exit(1)

    # ---- Interactive mode (no arguments) ----
    if not is_cli_mode:
        run_interactive_mode(db, storage)
        return

    # ---- CLI mode ----
    cmd = rest[0].lower() if rest else None

    # ---- Top-level commands (work without --group) ----

    # list — show available point groups
    if cmd == 'list':
        list_groups_cli(db)
        return

    # verify — verify one or all tables
    if cmd == 'verify':
        if args.group:
            verify_one(db, args.group)
        elif '--all' in rest:
            verify_all_tables(db)
        elif len(rest) > 1 and not rest[1].startswith('-'):
            verify_one(db, rest[1])
        else:
            print("Usage: verify [--all] [group_name]")
            print("  --all         Verify all tables")
            print("  group_name    Verify a specific group")
        return

    # storage — manage stored characters
    if cmd == 'storage':
        handle_storage_cli(db, storage, args.group, rest)
        return

    # ---- Group-specific commands (--group is now required) ----
    if not args.group:
        # Check if the first token looks like a group name
        if rest[0] in db.list_groups():
            print(f"Error: Use --group (or -g) to specify a group.")
            print(f"  Correct: python main.py -g {rest[0]} \"expression\"")
            print(f"  Or:      python main.py -g {rest[0]} table")
        elif rest[0] in ('ir', 'raman', 'vib', 'table', 'verify', 'IR', 'Raman',
                         'Vib', 'Table', 'Verify'):
            print(f"Error: --group/-g is required for '{rest[0]}'")
            print(f"  Correct: python main.py -g GROUP {rest[0]}")
        else:
            print(f"Error: Unknown command or missing --group flag.")
            print(f"  Usage: python main.py -g GROUP \"{rest[0]}\"")
            print(f"  Or:    python main.py list")
        sys.exit(1)

    # --group IS set — handle group-specific commands
    if cmd in ('ir', 'raman', 'IR', 'Raman'):
        handle_group_cmd(db, storage, args.group, cmd.lower(), args.json)
        return

    if cmd in ('vib', 'Vib'):
        # The rest after 'vib' are fixed atom counts
        vib_args = rest[1:]
        handle_vibration(db, storage, args.group, vib_args)
        return

    if cmd == 'table':
        handle_group_cmd(db, storage, args.group, 'table', args.json)
        return

    if cmd == 'verify':
        handle_group_cmd(db, storage, args.group, 'verify', args.json)
        return

    # ---- Expression evaluation ----
    expr_str = ' '.join(rest)
    try:
        evaluate_expression(db, storage, args.group, expr_str, args.json)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
