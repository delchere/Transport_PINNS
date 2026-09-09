# transport_pinns/utils.py
"""
Utility functions for the transport_pinns package
"""

import numpy as np
import os
from datetime import datetime

def create_output_dir(base_dir="output"):
    """Creates an output directory with a timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{base_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def save_matrix(matrix, filename, format='npy'):
    """Saves a matrix in different formats"""
    if format == 'npy':
        np.save(filename, matrix)
    elif format == 'txt':
        np.savetxt(filename, matrix)
    else:
        raise ValueError(f"Unsupported format: {format}")

def load_matrix(filename, format='auto'):
    """Loads a matrix from a file"""
    if format == 'auto':
        if filename.endswith('.npy'):
            return np.load(filename)
        elif filename.endswith('.txt') or filename.endswith('.dat'):
            return np.loadtxt(filename)
        else:
            raise ValueError("Unrecognized file format")
    elif format == 'npy':
        return np.load(filename)
    elif format == 'txt':
        return np.loadtxt(filename)

def print_matrix_info(matrix, name="Matrix"):
    """Prints information about a matrix"""
    print(f"{name}:")
    print(f"  Shape: {matrix.shape}")
    print(f"  Type: {matrix.dtype}")
    print(f"  Min: {np.min(matrix):.6e}")
    print(f"  Max: {np.max(matrix):.6e}")
    print(f"  Norm: {np.linalg.norm(matrix):.6e}")

