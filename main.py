"""
Main entry point - integrates all modules
"""

from character_table_database import CharacterTableDatabase
from character_calculator import CharacterCalculator
from calculator_ui import CalculatorUI
from character_storage import CharacterStorage
from constants import __version__, __author__, __license__, CATEGORY_ORDER


def display_welcome():
    """Display welcome message"""
    print("\n" + "=" * 80)
    print("Character Table Decomposer")
    print("=" * 80)
    print(f"Version: {__version__}")
    print(f"Author: {__author__}")
    print(f"License: {__license__}")
    print("\nFeatures:")
    print("  ✓ 30+ point groups")
    print("  ✓ Character decomposition")
    print("  ✓ Tensor products & direct sums")
    print("  ✓ Symmetric & antisymmetric powers")
    print("  ✓ Spherical harmonics / Atomic orbitals")
    print("  ✓ Polynomials (Sym^n)")
    print("  ✓ Power characters χ(g^n)")
    print("  ✓ Character storage to JSON")
    print("  ✓ Table verification")
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
    
    # Display by category order
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
    """Verify all character tables"""
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


def main():
    """Main function"""
    display_welcome()
    
    # Initialize
    print("\nLoading character tables...")
    try:
        db = CharacterTableDatabase()
        storage = CharacterStorage()
        print(f"✓ Loaded {len(db.list_groups())} point groups")
    except Exception as e:
        print(f"✗ Error: {e}")
        return
    
    # Main loop
    while True:
        try:
            display_map = display_groups(db)
            
            print(f"\n  V. Verify all tables")
            print(f"  0. Exit")
            
            choice = input(f"\nSelect group (1-{len(display_map)}, 0 to exit): ").strip()
            
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


def run_group_session(db: CharacterTableDatabase, storage: CharacterStorage, group_name: str):
    """Run interactive session for a group"""
    try:
        table = db.get_table(group_name)
        calculator = CharacterCalculator(table)
        ui = CalculatorUI(calculator, storage)
        
        ui.run_interactive_session()
    except Exception as e:
        print(f"✗ Error: {e}")


if __name__ == "__main__":
    main()