"""
Hamiltonian loader module.
Handles loading of Hamiltonian matrices and related data from files.
"""

import numpy as np
import tensorflow as tf
import os
from typing import Dict, Tuple, Optional, List
from pathlib import Path


class HamiltonianLoader:
    """
    Load Hamiltonian matrices and related data from files.
    Supports both .npy and .txt formats.
    """
    
    def __init__(self, config):
        """
        Initialize loader with configuration.
        
        Args:
            config: SimulationConfig object with path information
        """
        self.config = config
        self.paths = config.paths
        
        # Storage for loaded data
        self.data = {}
    
    def load_all(self) -> Dict:
        """
        Load all Hamiltonian data files.
        
        Returns:
            Dictionary containing all loaded matrices
        """
        self.data['H_0'] = self.load_matrix(self.paths.H_path)
        self.data['SELF'] = self.load_matrix(self.paths.sigma_path)
        self.data['Gamma'] = self.load_matrix(self.paths.gamma_path)
        #self.data['S'] = self.load_matrix(self.paths.S_path)
        
        # Try to load additional files if they exist
        self._load_additional_files()
        
        # Validate dimensions
        self._validate_dimensions()
        
        return self.data
    
    def load_matrix(self, filepath: str) -> np.ndarray:
        """
        Load a matrix from .npy or .txt file.
        
        Args:
            filepath: Path to the file
            
        Returns:
            Matrix as complex numpy array
        """
        filepath = str(filepath)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if filepath.endswith('.npy'):
            matrix = np.load(filepath)
        elif filepath.endswith('.txt'):
            matrix = np.loadtxt(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")
        
        # Ensure complex type
        if matrix.dtype != np.complex128 and matrix.dtype != np.complex64:
            matrix = matrix.astype(complex)
        
        print(f"✅ Loaded {os.path.basename(filepath)}: shape {matrix.shape}, dtype {matrix.dtype}")
        
        return matrix
    
    def _load_additional_files(self):
        """Load additional optional files if they exist."""
        base_dir = Path(self.paths.base_dir)
        
        # Try to load lead Hamiltonians
        h_lead_path = base_dir / "H_lead.npy"
        if h_lead_path.exists():
            self.data['H_lead'] = self.load_matrix(str(h_lead_path))
        
        # Try to load Fermi levels
        fermi_path = base_dir / "fermi_levels.txt"
        if fermi_path.exists():
            self.data['fermi_levels'] = np.loadtxt(str(fermi_path))
    
    def _validate_dimensions(self):
        """Validate that all matrices have consistent dimensions."""
        n = None
        
        for name, matrix in self.data.items():
            if name in ['H_0', 'SELF', 'Gamma','H_lead']:
                if len(matrix.shape) == 2 and matrix.shape[0] == matrix.shape[1]:
                    if n is None:
                        n = matrix.shape[0]
                        self.n = n
                    elif matrix.shape[0] != n:
                        print(f"⚠️ Warning: {name} has shape {matrix.shape}, expected ({n}, {n})")
        
        print(f"✅ System dimension: {self.n} orbitals")
    
    def get_system_dimension(self) -> int:
        """Get the dimension of the Hilbert space."""
        return self.n
    
    def to_tensorflow(self, dtype=tf.complex64) -> Dict:
        """
        Convert loaded matrices to TensorFlow tensors.
        
        Args:
            dtype: TensorFlow complex dtype
            
        Returns:
            Dictionary with TensorFlow tensors
        """
        tf_data = {}
        for key, value in self.data.items():
            if isinstance(value, np.ndarray):
                tf_data[key] = tf.cast(value, dtype)
        return tf_data
    
    def print_summary(self):
        """Print summary of loaded data."""
        print("\n" + "="*60)
        print("HAMILTONIAN LOADER SUMMARY")
        print("="*60)
        
        for name, matrix in self.data.items():
            if isinstance(matrix, np.ndarray):
                shape_str = f"{matrix.shape}"
                dtype_str = f"{matrix.dtype}"
                if len(matrix.shape) == 2 and matrix.shape[0] == matrix.shape[1]:
                    is_herm = np.allclose(matrix, matrix.conj().T, atol=1e-8)
                    herm_str = "Hermitian ✓" if is_herm else "Non-Hermitian"
                else:
                    herm_str = ""
                print(f"  {name:12s}: shape {shape_str:15s} {dtype_str:12s} {herm_str}")
        
        print("="*60)
