"""
Character storage and management system
Handles saving/loading stored characters to JSON
"""

import json
import os
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class CharacterStorage:
    """Manages character storage and persistence"""
    
    def __init__(self, storage_file: str = 'stored_characters.json'):
        """
        Initialize character storage
        
        Args:
            storage_file: JSON file path for persistent storage
        """
        self.storage_file = storage_file
        self.stored_characters: Dict[str, Dict[str, Dict]] = {}
        self.load_characters()
    
    def store_character(self, group_name: str, char_name: str, 
                       character: List, description: str = "") -> None:
        """
        Store a character in memory and to disk
        
        Args:
            group_name: Point group name (e.g., 'O_h')
            char_name: Name for this character
            character: List of character values
            description: Optional description
        """
        if group_name not in self.stored_characters:
            self.stored_characters[group_name] = {}
        
        self.stored_characters[group_name][char_name] = {
            'character': character,
            'description': description
        }
        
        self.save_characters()
        print(f"✓ Character '{char_name}' stored in {group_name}")
    
    def get_character(self, group_name: str, char_name: str) -> Optional[Tuple[List, str]]:
        """
        Retrieve a stored character
        
        Args:
            group_name: Point group name
            char_name: Character name
            
        Returns:
            Tuple of (character vector, description) or None
        """
        if (group_name in self.stored_characters and 
            char_name in self.stored_characters[group_name]):
            char_info = self.stored_characters[group_name][char_name]
            return char_info['character'], char_info['description']
        return None
    
    def list_stored_characters(self, group_name: str) -> List[str]:
        """
        List all stored characters for a group
        
        Args:
            group_name: Point group name
            
        Returns:
            List of character names
        """
        if group_name in self.stored_characters:
            return list(self.stored_characters[group_name].keys())
        return []
    
    def delete_character(self, group_name: str, char_name: str) -> bool:
        """
        Delete a stored character
        
        Args:
            group_name: Point group name
            char_name: Character name
            
        Returns:
            True if deleted, False if not found
        """
        if (group_name in self.stored_characters and 
            char_name in self.stored_characters[group_name]):
            del self.stored_characters[group_name][char_name]
            self.save_characters()
            print(f"✓ Character '{char_name}' deleted from {group_name}")
            return True
        return False
    
    def save_characters(self) -> None:
        """Save all characters to JSON file"""
        try:
            serializable = {}
            for group_name, chars in self.stored_characters.items():
                serializable[group_name] = {}
                for char_name, char_info in chars.items():
                    # Convert numpy types to Python native types
                    character = []
                    for x in char_info['character']:
                        if isinstance(x, complex):
                            character.append(str(x))
                        else:
                            character.append(float(x))
                    
                    serializable[group_name][char_name] = {
                        'character': character,
                        'description': char_info['description']
                    }
            
            with open(self.storage_file, 'w', encoding='utf-8') as f:
                json.dump(serializable, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"✗ Error saving characters: {e}")
    
    def load_characters(self) -> None:
        """Load characters from JSON file"""
        try:
            if os.path.exists(self.storage_file):
                with open(self.storage_file, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                
                self.stored_characters = {}
                for group_name, chars in loaded.items():
                    self.stored_characters[group_name] = {}
                    for char_name, char_info in chars.items():
                        character = []
                        for val in char_info['character']:
                            try:
                                if isinstance(val, str) and ('j' in val or 'i' in val):
                                    val = val.replace('i', 'j')
                                    character.append(complex(val))
                                else:
                                    character.append(float(val))
                            except ValueError:
                                character.append(float(val))
                        
                        self.stored_characters[group_name][char_name] = {
                            'character': character,
                            'description': char_info['description']
                        }
                
                total = sum(len(chars) for chars in self.stored_characters.values())
                print(f"✓ Loaded {total} stored characters")
        except Exception as e:
            print(f"✗ Error loading characters: {e}")
    
    def export_to_csv(self, group_name: str, char_name: str, filename: str) -> bool:
        """Export a character to CSV"""
        char_tuple = self.get_character(group_name, char_name)
        if not char_tuple:
            return False
        
        character, description = char_tuple
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Description: {description}\n")
                f.write("Values:\n")
                for i, val in enumerate(character):
                    f.write(f"{i},{val}\n")
            print(f"✓ Exported to {filename}")
            return True
        except Exception as e:
            print(f"✗ Error exporting: {e}")
            return False