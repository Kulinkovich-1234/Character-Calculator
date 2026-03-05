"""
Interactive user interface with full feature support
Imports: character_calculator, character_storage
"""

from typing import Optional, List, Tuple
from character_calculator import CharacterCalculator
from character_storage import CharacterStorage


class CalculatorUI:
    """Handles all user interaction with 5 input methods"""
    
    def __init__(self, calculator: CharacterCalculator, storage: CharacterStorage):
        """
        Initialize UI
        
        Args:
            calculator: CharacterCalculator instance
            storage: CharacterStorage instance
        """
        self.calc = calculator
        self.storage = storage
    
    def run_interactive_session(self):
        """Main interactive loop"""
        self.print_character_table()
        
        while True:
            print("\n" + "=" * 80)
            print(f"Character Table Calculator: {self.calc.table.name}")
            print(f"Order: {self.calc.group_order} | Classes: {len(self.calc.class_sizes)}")
            print("=" * 80)
            
            self._show_menu()
            choice = input("\nSelect option: ").strip()
            
            if choice == "0":
                print("Returning to group selection...")
                break
            elif choice == "1":
                self._decompose_character()
            elif choice == "2":
                self._tensor_product()
            elif choice == "3":
                self._symmetric_products()
            elif choice == "4":
                self._direct_sum()
            elif choice == "5":
                self._power_character()
            elif choice == "6":
                self._manage_storage()
            elif choice == "7":
                self.print_character_table()
            elif choice == "8":
                self._verify_table()
            elif choice == "9":
                self._conjugate_classes_sn()
            else:
                print("✗ Invalid choice")
    
    def _show_menu(self):
        """Display menu options"""
        print("\nOptions:")
        print("  1. Decompose character")
        print("  2. Tensor product")
        print("  3. Symmetric/antisymmetric products")
        print("  4. Direct sum")
        print("  5. Power of character χ(g^n)")
        print("  6. Manage stored characters")
        print("  7. View character table")
        print("  8. Verify character table")
        print("  9. Conjugacy classes of S_n")
        print("  0. Exit")
    
    # ==================== Main Operations ====================
    
    def _decompose_character(self):
        """Handle character decomposition"""
        char, source, name = self.input_character("Decomposition")
        if char is None:
            return
        
        try:
            decomp = self.calc.decompose(char)
            decomp_str = self.calc.format_decomposition(decomp)
            
            print(f"\n✓ {name} ({source}):")
            print(f"  Decomposition: {decomp_str}")
            
            self._ask_to_store(char, source, name)
        except ValueError as e:
            print(f"✗ Error: {e}")
    
    def _tensor_product(self):
        """Handle tensor product calculation"""
        print("\nTensor Product χ₁ ⊗ χ₂")
        
        char1, source1, name1 = self.input_character("First character")
        if char1 is None:
            return
        
        char2, source2, name2 = self.input_character("Second character")
        if char2 is None:
            return
        
        try:
            tensor_char, decomp = self.calc.tensor_product(char1, char2)
            decomp_str = self.calc.format_decomposition(decomp)
            
            print(f"\n✓ {name1} ⊗ {name2}:")
            print(f"  Decomposition: {decomp_str}")
            
            self._ask_to_store(tensor_char, 'tensor_product', f"{name1}⊗{name2}")
        except ValueError as e:
            print(f"✗ Error: {e}")
    
    def _symmetric_products(self):
        """Handle symmetric/antisymmetric products"""
        n = self._get_positive_int("Power (n >= 0): ")
        if n is None:
            return
        
        char, source, name = self.input_character("Character")
        if char is None:
            return
        
        try:
            sym_char, sym_decomp, antisym_char, antisym_decomp = \
                self.calc.symmetric_and_antisymmetric_products(char, n)
            
            sym_str = self.calc.format_decomposition(sym_decomp)
            antisym_str = self.calc.format_decomposition(antisym_decomp)
            
            print(f"\n✓ {name} (n={n}):")
            print(f"  Sym^{n}: {sym_str}")
            print(f"  Alt^{n}: {antisym_str}")
            
            self._ask_to_store(sym_char, 'symmetric_product', f"Sym^{n}({name})")
            self._ask_to_store(antisym_char, 'antisymmetric_product', f"Alt^{n}({name})")
        except ValueError as e:
            print(f"✗ Error: {e}")
    
    def _direct_sum(self):
        """Handle direct sum calculation"""
        print("\nDirect Sum χ₁ ⊕ χ₂")
        
        char1, source1, name1 = self.input_character("First character")
        if char1 is None:
            return
        
        char2, source2, name2 = self.input_character("Second character")
        if char2 is None:
            return
        
        try:
            direct_sum_char, decomp = self.calc.direct_sum(char1, char2)
            decomp_str = self.calc.format_decomposition(decomp)
            
            print(f"\n✓ {name1} ⊕ {name2}:")
            print(f"  Decomposition: {decomp_str}")
            
            self._ask_to_store(direct_sum_char, 'direct_sum', f"{name1}⊕{name2}")
        except ValueError as e:
            print(f"✗ Error: {e}")
    
    def _power_character(self):
        """Handle power of character χ(g^n)"""
        n = self._get_integer("Power (n): ")
        if n is None:
            return
        
        char, source, name = self.input_character("Character")
        if char is None:
            return
        
        try:
            power_char = self.calc.get_character_at_power(char, n)
            decomp = self.calc.decompose(power_char)
            decomp_str = self.calc.format_decomposition(decomp)
            
            print(f"\n✓ {name}^{n}:")
            print(f"  χ(g^{n}): {power_char}")
            print(f"  Decomposition: {decomp_str}")
            
            self._ask_to_store(power_char, 'power', f"{name}^{n}")
        except ValueError as e:
            print(f"✗ Error: {e}")
    
    def _manage_storage(self):
        """Manage stored characters"""
        group_name = self.calc.table.name
        
        while True:
            print(f"\n{group_name} Stored Characters")
            print("=" * 60)
            
            stored = self.storage.list_stored_characters(group_name)
            
            if not stored:
                print("No stored characters for this group")
                break
            
            for i, name in enumerate(stored, 1):
                char_tuple = self.storage.get_character(group_name, name)
                if char_tuple:
                    _, desc = char_tuple
                    print(f"{i}. {name}")
                    if desc:
                        print(f"   {desc}")
            
            print("\nOptions:")
            print("  1. View character")
            print("  2. Delete character")
            print("  3. Export to CSV")
            print("  0. Return")
            
            choice = input("\nSelect: ").strip()
            
            if choice == "0":
                break
            elif choice == "1":
                self._view_stored()
            elif choice == "2":
                self._delete_stored(stored)
            elif choice == "3":
                self._export_stored()
    
    def _verify_table(self):
        """Verify character table"""
        print("\nVerifying character table...")
        result = self.calc.verify_table(verbose=True)
        
        if result:
            print(f"\n✓ All checks passed!")
        else:
            print(f"\n✗ Some checks failed")
    
    def _conjugate_classes_sn(self):
        """Show conjugacy classes of S_n"""
        n = self._get_positive_int("n for S_n: ")
        if n is None:
            return
        
        print(f"\nConjugacy Classes of S_{n}:")
        print("=" * 80)
        print(f"{'Partition':<20} {'Size':<15} {'Sign':<10}")
        print("-" * 80)
        
        classes = CharacterCalculator._conjugate_classes_sn(n)
        
        for partition_desc, partition, size, sign in classes:
            sign_str = f"({sign:+d})" if sign != 0 else "(0)"
            # Now partition_desc is a string, so formatting works
            print(f"{partition_desc:<20} {size:<15} {sign_str:<10}")
        
        print("-" * 80)
        total = sum(size for _, _, size, _ in classes)
        print(f"Total permutations: {total} = {n}!")
    
    # ==================== Input Methods (5 formats) ====================
    
    def input_character(self, prompt: str = "Select input method") -> Tuple[Optional[List], str, str]:
        """
        Interactive character input with 5 methods:
        1. Manual input (type numbers)
        2. Stored character (load from JSON)
        3. Irreducible representation (from table)
        4. Spherical harmonics / Atomic orbital (l or letter)
        5. Polynomial (symmetric power)
        
        Returns:
            Tuple of (character, source_type, name)
        """
        group_name = self.calc.table.name
        
        print(f"\n{prompt}:")
        print("  1. Manual input")
        print("  2. Stored character")
        print("  3. Irreducible representation")
        print("  4. Spherical harmonics / Atomic orbital")
        print("  5. Polynomial (Sym^n)")
        
        choice = input("Select method (1-5): ").strip()
        
        if choice == "1":
            return self._input_manual()
        elif choice == "2":
            return self._input_stored(group_name)
        elif choice == "3":
            return self._input_irrep()
        elif choice == "4":
            return self._input_spherical_harmonics()
        elif choice == "5":
            return self._input_polynomial()
        else:
            print("✗ Invalid choice")
            return None, None, None
    
    def _input_manual(self) -> Tuple[Optional[List], str, str]:
        """Method 1: Manual input"""
        print(f"\nManual Input for {self.calc.table.name}")
        print(f"Classes: {self.calc.class_names}")
        print(f"Class sizes: {self.calc.class_sizes}")
        
        while True:
            user_input = input(
                f"Enter {len(self.calc.class_sizes)} numbers (space-separated, or 'q' to cancel): "
            ).strip()
            
            if user_input.lower() == 'q':
                return None, None, None
            
            try:
                values = []
                for val_str in user_input.split():
                    if 'j' in val_str or 'i' in val_str:
                        val_str = val_str.replace('i', 'j')
                        values.append(complex(val_str))
                    else:
                        values.append(float(val_str))
                
                if len(values) != len(self.calc.class_sizes):
                    print(f"✗ Expected {len(self.calc.class_sizes)} values")
                    continue
                
                return values, 'manual', 'Manual input'
                
            except ValueError:
                print("✗ Invalid format")
    
    def _input_stored(self, group_name: str) -> Tuple[Optional[List], str, str]:
        """Method 2: Stored character"""
        stored = self.storage.list_stored_characters(group_name)
        
        if not stored:
            print("✗ No stored characters for this group")
            return None, None, None
        
        print("\nStored Characters:")
        for i, name in enumerate(stored, 1):
            print(f"  {i}. {name}")
        
        choice = input("Select (number or name): ").strip()
        
        try:
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(stored):
                    name = stored[idx]
                else:
                    print("✗ Invalid index")
                    return None, None, None
            else:
                if choice in stored:
                    name = choice
                else:
                    print("✗ Not found")
                    return None, None, None
            
            char_tuple = self.storage.get_character(group_name, name)
            if char_tuple:
                char, desc = char_tuple
                return char, 'stored', name
            return None, None, None
            
        except ValueError:
            print("✗ Invalid input")
            return None, None, None
    
    def _input_irrep(self) -> Tuple[Optional[List], str, str]:
        """Method 3: Irreducible representation"""
        print("\nIrreducible Representations:")
        irreps_list = list(self.calc.irreps.keys())
        
        for i, name in enumerate(irreps_list, 1):
            print(f"  {i}. {name}")
        
        choice = input("Select (number or name): ").strip()
        
        try:
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(irreps_list):
                    name = irreps_list[idx]
                else:
                    print("✗ Invalid index")
                    return None, None, None
            else:
                if choice in self.calc.irreps:
                    name = choice
                else:
                    print("✗ Not found")
                    return None, None, None
            
            char = self.calc.irreps[name]
            return list(char), 'irrep', name
            
        except ValueError:
            print("✗ Invalid input")
            return None, None, None
    
    def _input_spherical_harmonics(self) -> Tuple[Optional[List], str, str]:
        """Method 4: Spherical harmonics / Atomic orbital"""
        if self.calc.vector_char is None:
            print(f"✗ Vector representation not defined for {self.calc.table.name}")
            return None, None, None
        
        print("\nSpherical Harmonics / Atomic Orbital")
        print("Enter angular quantum number l or orbital letter (s,p,d,f,...):")
        
        while True:
            orbital_input = input("Input (or 'q' to cancel): ").strip()
            
            if orbital_input.lower() == 'q':
                return None, None, None
            
            try:
                l = CharacterCalculator.parse_orbital_input(orbital_input)
                char = self.calc.harmonic_character(l)
                
                # Name
                if orbital_input.lower() in ['s', 'p', 'd', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o']:
                    name = orbital_input.lower()
                else:
                    name = f"l={l}"
                
                return char, 'spherical_harmonics', name
                
            except ValueError as e:
                print(f"✗ {e}")
    
    def _input_polynomial(self) -> Tuple[Optional[List], str, str]:
        """Method 5: Polynomial (Sym^n)"""
        if self.calc.vector_char is None:
            print(f"✗ Vector representation not defined for {self.calc.table.name}")
            return None, None, None
        
        print("\nPolynomial (Symmetric Power)")
        print("Enter degree n of homogeneous polynomial (Sym^n):")
        
        while True:
            n_input = input("Input (or 'q' to cancel): ").strip()
            
            if n_input.lower() == 'q':
                return None, None, None
            
            try:
                n = int(n_input)
                if n < 0:
                    print("✗ Must be non-negative")
                    continue
                
                char = self.calc.polynomial_character(n)
                name = f"P_{n}"
                
                return char, 'polynomial', name
                
            except ValueError:
                print("✗ Invalid integer")
    
    # ==================== Helper Methods ====================
    
    def _ask_to_store(self, character: List, source_type: str, source_name: str) -> bool:
        """Ask user to store a character"""
        store = input(f"\nStore result? (y/n): ").strip().lower()
        if store not in ('y', 'yes'):
            return False
        
        group_name = self.calc.table.name
        default_name = self._get_default_name(source_type, source_name)
        
        name = input(f"Name (default: '{default_name}'): ").strip()
        if not name:
            name = default_name
        
        # Check if exists
        existing = self.storage.get_character(group_name, name)
        if existing:
            overwrite = input(f"Character '{name}' exists. Overwrite? (y/n): ").strip().lower()
            if overwrite not in ('y', 'yes'):
                return False
        
        description = input("Description (optional): ").strip()
        self.storage.store_character(group_name, name, character, description)
        return True
    
    def _get_default_name(self, source_type: str, source_name: str) -> str:
        """Generate default name based on source"""
        if source_type == 'irrep':
            return f"{source_name}_char"
        elif source_type == 'stored':
            return f"{source_name}_copy"
        elif source_type == 'spherical_harmonics':
            return f"{source_name}_orbital"
        elif source_type == 'polynomial':
            return f"{source_name}_poly"
        else:
            return "custom_char"
    
    def _get_positive_int(self, prompt: str) -> Optional[int]:
        """Get non-negative integer"""
        while True:
            try:
                val = int(input(prompt).strip())
                if val >= 0:
                    return val
                print("✗ Must be non-negative")
            except ValueError:
                print("✗ Invalid integer")
            except KeyboardInterrupt:
                return None
    
    def _get_integer(self, prompt: str) -> Optional[int]:
        """Get any integer"""
        while True:
            try:
                return int(input(prompt).strip())
            except ValueError:
                print("✗ Invalid integer")
            except KeyboardInterrupt:
                return None
    
    def _view_stored(self):
        """View a stored character"""
        group_name = self.calc.table.name
        stored = self.storage.list_stored_characters(group_name)
        
        choice = input("Character name: ").strip()
        if choice not in stored:
            print("✗ Not found")
            return
        
        char_tuple = self.storage.get_character(group_name, choice)
        if char_tuple:
            char, desc = char_tuple
            print(f"\n{choice}:")
            if desc:
                print(f"Description: {desc}")
            print(f"Character: {char}")
    
    def _delete_stored(self, stored: List[str]):
        """Delete a stored character"""
        group_name = self.calc.table.name
        choice = input("Character name: ").strip()
        if choice not in stored:
            print("✗ Not found")
            return
        
        confirm = input(f"Delete '{choice}'? (y/n): ").strip().lower()
        if confirm in ('y', 'yes'):
            self.storage.delete_character(group_name, choice)
    
    def _export_stored(self):
        """Export a stored character to CSV"""
        group_name = self.calc.table.name
        stored = self.storage.list_stored_characters(group_name)
        
        choice = input("Character name: ").strip()
        if choice not in stored:
            print("✗ Not found")
            return
        
        filename = input("Filename (default: '{}.csv'): ".format(choice)).strip()
        if not filename:
            filename = f"{choice}.csv"
        
        self.storage.export_to_csv(group_name, choice, filename)
    
    def print_character_table(self):
        """Display character table"""
        self.calc.print_character_table()