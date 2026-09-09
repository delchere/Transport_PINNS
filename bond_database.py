"""
Typical Bond Length Database with Built-in Storage
"""

import numpy as np
import os
from typing import Dict, Optional, List, Tuple
from datetime import datetime

class BondDatabase:
    """Database of typical chemical bond lengths"""
    
    def __init__(self):
        # Extended database of typical bond lengths (in Å)
        self.bond_lengths_db = {
            # Noble metals
            'Pt-Pt': 2.41, 'Pt-H': 1.70, 'Pt-C': 1.92, 'Pt-N': 1.95, 'Pt-O': 1.98, 'Pt-S': 2.30,
            'Au-Au': 2.88, 'Au-H': 1.60, 'Au-C': 2.05, 'Au-N': 2.10, 'Au-O': 2.10, 'Au-S': 2.30,
            'Ag-Ag': 2.89, 'Ag-H': 1.65, 'Ag-C': 2.20, 'Ag-N': 2.25, 'Ag-O': 2.25, 'Ag-S': 2.45,
            'Cu-Cu': 2.56, 'Cu-H': 1.60, 'Cu-C': 1.92, 'Cu-N': 1.97, 'Cu-O': 1.96, 'Cu-S': 2.20,
            
            # Transition metals
            'Fe-Fe': 2.48, 'Fe-H': 1.56, 'Fe-C': 1.92, 'Fe-N': 1.96, 'Fe-O': 1.96, 'Fe-S': 2.25,
            'Ni-Ni': 2.49, 'Ni-H': 1.47, 'Ni-C': 1.82, 'Ni-N': 1.86, 'Ni-O': 1.84, 'Ni-S': 2.20,
            'Co-Co': 2.51, 'Co-H': 1.52, 'Co-C': 1.90, 'Co-N': 1.94, 'Co-O': 1.93, 'Co-S': 2.22,
            'Pd-Pd': 2.75, 'Pd-H': 1.65, 'Pd-C': 2.02, 'Pd-N': 2.05, 'Pd-O': 2.07, 'Pd-S': 2.32,
            'Rh-Rh': 2.69, 'Rh-H': 1.60, 'Rh-C': 2.00, 'Rh-N': 2.03, 'Rh-O': 2.05, 'Rh-S': 2.30,
            
            # Carbon family elements
            'C-C': 1.42, 'C-H': 1.09, 'C-N': 1.47, 'C-O': 1.43, 'C-S': 1.82, 'C-F': 1.35, 'C-Cl': 1.77,
            'Si-Si': 2.35, 'Si-H': 1.48, 'Si-C': 1.89, 'Si-N': 1.74, 'Si-O': 1.63, 'Si-S': 2.15,
            
            # Common elements
            'H-H': 0.90, 'H-N': 1.01, 'H-O': 0.96, 'H-S': 1.34, 'H-F': 0.92, 'H-Cl': 1.27,
            'N-N': 1.45, 'N-O': 1.40, 'N-S': 1.65, 'N-F': 1.36, 'N-Cl': 1.75,
            'O-O': 1.48, 'O-S': 1.50, 'O-F': 1.42, 'O-Cl': 1.70,
            'S-S': 2.05, 'S-F': 1.60, 'S-Cl': 2.00,
        }
        
        # Default values for missing bonds
        self.default_bonds = {
            'C-C': 1.54,  # single bond
            'C=C': 1.34,  # double bond
            'C≡C': 1.20,  # triple bond
            'C-H': 1.09,
            'O-H': 0.96,
            'N-H': 1.01,
        }
        
        # Output folder configuration
        self.output_dir = "bond_database_output"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def get_bond_length(self, element1: str, element2: str, 
                       bond_order: int = 1) -> Optional[float]:
        """Returns the bond length for a pair of elements"""
        # Try with specified bond order
        if bond_order > 1:
            bond_type = f"{element1}={element2}" if bond_order == 2 else f"{element1}≡{element2}"
            if bond_type in self.bond_lengths_db:
                return self.bond_lengths_db[bond_type]
        
        # Try standard versions
        bond_type1 = f"{element1}-{element2}"
        bond_type2 = f"{element2}-{element1}"
        
        if bond_type1 in self.bond_lengths_db:
            return self.bond_lengths_db[bond_type1]
        elif bond_type2 in self.bond_lengths_db:
            return self.bond_lengths_db[bond_type2]
        else:
            # Check in default values
            default_key = f"{element1}-{element2}" if element1 <= element2 else f"{element2}-{element1}"
            return self.default_bonds.get(default_key, None)
    
    def add_bond_length(self, element1: str, element2: str, length: float,
                       bond_order: int = 1):
        """Adds a new bond length to the database"""
        if bond_order == 1:
            bond_type = f"{element1}-{element2}"
        elif bond_order == 2:
            bond_type = f"{element1}={element2}"
        elif bond_order == 3:
            bond_type = f"{element1}≡{element2}"
        else:
            bond_type = f"{element1}-{element2}"
        
        self.bond_lengths_db[bond_type] = length
    
    def get_required_bonds(self, symbols: str, save_to_file: bool = True) -> Dict[str, float]:
        """
        Determines the required bonds for a sequence of atoms
        and automatically saves the results
        """
        from ase import Atoms
        
        temp_atoms = Atoms(symbols)
        atom_sequence = list(temp_atoms.symbols)
        
        required_bonds = {}
        
        print(f"\n🔍 Analyzing structure: {symbols}")
        print(f"   Sequence: {'-'.join(atom_sequence)}")
        
        # Identify all adjacent pairs
        for i in range(len(atom_sequence) - 1):
            el1, el2 = atom_sequence[i], atom_sequence[i + 1]
            bond_type = f"{el1}-{el2}"
            bond_length = self.get_bond_length(el1, el2)
            
            if bond_length is not None:
                required_bonds[bond_type] = bond_length
                print(f"   ✓ {bond_type}: {bond_length} Å")
            else:
                # Ask the user
                print(f"   ⚠️  Unknown bond length for: {bond_type}")
                try:
                    length = float(input(f"   Enter the length for {bond_type} (Å): "))
                    required_bonds[bond_type] = length
                    self.add_bond_length(el1, el2, length)
                except:
                    # Use a reasonable default
                    default_length = 1.5  # Å
                    print(f"     Using default: {default_length} Å")
                    required_bonds[bond_type] = default_length
        
        # Also identify intra-group bonds
        elements = list(set(atom_sequence))
        for el in elements:
            bond_type = f"{el}-{el}"
            bond_length = self.get_bond_length(el, el)
            if bond_length is not None and bond_type not in required_bonds:
                required_bonds[bond_type] = bond_length
        
        # Save results if requested
        if save_to_file:
            self._save_results(symbols, required_bonds, atom_sequence)
        
        return required_bonds
    
    def _save_results(self, formula: str, bonds: Dict[str, float], atom_sequence: List[str]):
        """Saves results to a file - INTERNAL to the module"""
        # Create filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"bonds_{formula}_{timestamp}.txt"
        filepath = os.path.join(self.output_dir, filename)
        
        # Write to file
        with open(filepath, 'w') as f:
            f.write(f"Bond lengths for: {formula}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Atom sequence: {'-'.join(atom_sequence)}\n\n")
            f.write("REQUIRED BONDS:\n")
            f.write("-" * 30 + "\n")
            
            for bond, length in bonds.items():
                f.write(f"{bond}: {length} Å\n")
        
        print(f"\n💾 Results saved in: {filepath}")
    
    def suggest_bond_length(self, element1: str, element2: str) -> float:
        """Suggests a reasonable bond length"""
        length = self.get_bond_length(element1, element2)
        if length is not None:
            return length
        
        # Pauling rules for estimation
        covalent_radii = {
            'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57,
            'Si': 1.11, 'P': 1.07, 'S': 1.05, 'Cl': 1.02,
            'Pt': 1.36, 'Au': 1.36, 'Ag': 1.45, 'Cu': 1.32,
        }
        
        r1 = covalent_radii.get(element1, 1.5)
        r2 = covalent_radii.get(element2, 1.5)
        
        return r1 + r2
    
    def list_available_bonds(self, element: str = None) -> List[Tuple[str, float]]:
        """Lists all available bonds, optionally filtered by element"""
        bonds = []
        for bond_type, length in self.bond_lengths_db.items():
            if element is None or element in bond_type:
                bonds.append((bond_type, length))
        return sorted(bonds)

