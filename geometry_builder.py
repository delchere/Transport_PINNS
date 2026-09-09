# geometry_builder.py
"""
Molecular Geometry Builder with Visualization
"""

import numpy as np
from ase import Atoms
import os
from datetime import datetime
import matplotlib.pyplot as plt

class GeometryBuilder:
    """Molecular geometry builder with visualization"""
    
    def __init__(self, bond_db):
        self.bond_db = bond_db
        self.output_dir = "geometry_output"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def build_molecule(self, symbols: str, L: float = 7.0, visualize: bool = True):
        """
        Builds a molecule and generates a visualization.
        
        Args:
            symbols: Chemical formula (e.g., "Pt2H2Pt2")
            L: Cell width (Å)
            visualize: If True, generates a plot of the atoms
        """
        print(f"\n🔨 Building molecule: {symbols}")
        
        # Get bond lengths
        bond_lengths = self.bond_db.get_required_bonds(symbols)
        
        # Parse the formula
        temp_atoms = Atoms(symbols)
        atom_sequence = list(temp_atoms.symbols)
        
        print(f"   Sequence: {'-'.join(atom_sequence)}")
        
        # Build positions
        positions = []
        current_x = 0.0
        
        for i in range(len(atom_sequence)):
            element = atom_sequence[i]
            
            if i > 0:
                prev_element = atom_sequence[i-1]
                bond_key = f"{prev_element}-{element}"
                bond_length = bond_lengths.get(bond_key, 1.5)
                current_x += bond_length
                print(f"   {bond_key}: {bond_length:.3f} Å")
            
            positions.append([current_x, L/2, L/2])
        
        # Create Atoms object
        total_length = positions[-1][0]
        cell_x = total_length + 2.0
        
        atoms = Atoms(
            symbols=atom_sequence,
            positions=positions,
            cell=[cell_x, L, L],
            pbc=[True, False, False]
        )
        
        print(f"\n✅ Created {len(atoms)} atoms")
        print(f"   Cell: {cell_x:.2f} × {L:.2f} × {L:.2f} Å")
        
        # Save geometry
        self._save_geometry(atoms, symbols)
        
        # Visualize if requested
        if visualize:
            self._visualize_atoms(atoms, symbols)
        
        return atoms
    
    def _save_geometry(self, atoms, formula):
        """Saves the geometry to files"""
        safe_name = formula.replace('(', '').replace(')', '').replace(' ', '_')
        
        # Save positions
        pos_file = os.path.join(self.output_dir, f"{safe_name}_positions.txt")
        np.savetxt(pos_file, atoms.positions, 
                  header=f"Positions (Å) for {formula}\nx y z")
        
        # Save info
        info_file = os.path.join(self.output_dir, f"{safe_name}_info.txt")
        with open(info_file, 'w') as f:
            f.write(f"Geometry: {formula}\n")
            f.write(f"Atoms: {len(atoms)}\n")
            f.write(f"Cell: {atoms.cell[0,0]:.2f} {atoms.cell[1,1]:.2f} {atoms.cell[2,2]:.2f} Å\n")
            f.write("\nCoordinates:\n")
            for i, (sym, pos) in enumerate(zip(atoms.symbols, atoms.positions)):
                f.write(f"{i:2d} {sym:3s} {pos[0]:8.3f} {pos[1]:8.3f} {pos[2]:8.3f}\n")
        
        print(f"   Files saved in: {self.output_dir}/")
    
    def _visualize_atoms(self, atoms, formula):
        """Generates a simple visualization of the atoms"""
        try:
            plt.figure(figsize=(10, 4))
            
            # Colors for elements
            colors = {'Pt': 'gold', 'H': 'white', 'C': 'black', 
                      'O': 'red', 'N': 'blue', 'Au': 'orange'}
            
            # Sizes for elements
            sizes = {'Pt': 300, 'H': 100, 'C': 200, 'O': 200, 'N': 200, 'Au': 300}
            
            # Plot each atom
            for i, (symbol, pos) in enumerate(zip(atoms.symbols, atoms.positions)):
                color = colors.get(symbol, 'gray')
                size = sizes.get(symbol, 200)
                
                plt.scatter(pos[0], pos[1], color=color, s=size, 
                            edgecolor='black', zorder=2)
                plt.text(pos[0], pos[1] + 0.2, symbol, 
                         ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            # Plot bonds
            for i in range(len(atoms) - 1):
                x1, y1, _ = atoms.positions[i]
                x2, y2, _ = atoms.positions[i + 1]
                plt.plot([x1, x2], [y1, y2], 'k-', linewidth=2, alpha=0.5, zorder=1)
                
                # Distance label
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                dist = atoms.get_distance(i, i+1)
                plt.text(mid_x, mid_y - 0.3, f'{dist:.2f}Å', 
                         ha='center', va='top', fontsize=8,
                         bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
            
            # Draw cell
            cell_x, cell_y, _ = atoms.cell.diagonal()
            plt.plot([0, cell_x, cell_x, 0, 0], 
                     [0, 0, cell_y, cell_y, 0], 'r--', alpha=0.5, label='Cell')
            
            plt.xlabel('X (Å)', fontsize=12)
            plt.ylabel('Y (Å)', fontsize=12)
            plt.title(f'Geometry: {formula}', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.axis('equal')
            
            # Save image
            safe_name = formula.replace('(', '').replace(')', '').replace(' ', '_')
            img_file = os.path.join(self.output_dir, f"{safe_name}_visualization.png")
            plt.savefig(img_file, dpi=150, bbox_inches='tight')
            print(f"   📊 Visualization saved: {img_file}")
            
            plt.show()
            
        except Exception as e:
            print(f"   ⚠️  Visualization error: {e}")

