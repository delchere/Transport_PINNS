"""
Observables module.
Computes physical observables like current with two different methods.
User must explicitly specify the orbitals in configuration.
"""

import tensorflow as tf
import numpy as np
from typing import List, Tuple, Dict, Union, Optional
from abc import ABC, abstractmethod


class CurrentCalculator(ABC):
    """
    Abstract base class for current calculators.
    """
    
    def __init__(self, system, params=None):
        """
        Initialize current calculator.
        
        Args:
            system: QuantumSystem instance
            params: Additional parameters
        """
        self.system = system
        self.params = params or {}
    
    @abstractmethod
    def compute_current(self, psi_total: tf.Tensor, HT_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute current from wavefunction and Hamiltonian.
        
        Args:
            psi_total: Total wavefunction tensor [N_time, n]
            HT_pred: Total Hamiltonian tensor [N_time, n, n]
            
        Returns:
            Current tensor [N_time]
        """
        pass
    
    @abstractmethod
    def get_info(self) -> Dict:
        """Return information about the current calculation method."""
        pass


class OrbitalPairCurrent(CurrentCalculator):
    """
    Current between two specific orbitals.
    I = ψ_i* × H_ij × ψ_j
    
    User MUST specify both orbital indices. No defaults.
    """
    
    def __init__(self, system, orbital_i: int, orbital_j: int, label: str = ""):
        """
        Initialize orbital pair current calculator.
        
        Args:
            system: QuantumSystem instance
            orbital_i: Index of first orbital (required)
            orbital_j: Index of second orbital (required)
            label: Optional label for the current
            
        Raises:
            ValueError: If orbital indices are not provided or out of range
        """
        super().__init__(system)
        
        # Explicitly require both indices
        if orbital_i is None or orbital_j is None:
            raise ValueError("Orbital indices orbital_i and orbital_j must be specified in config")
        
        self.orbital_i = orbital_i
        self.orbital_j = orbital_j
        self.label = label or f"I_{orbital_i}{orbital_j}"
        
        # Validate orbital indices
        n = system.n
        if orbital_i >= n or orbital_j >= n or orbital_i < 0 or orbital_j < 0:
            raise ValueError(f"Orbital indices must be between 0 and {n-1}, got ({orbital_i}, {orbital_j})")
    
    def compute_current(self, psi_total: tf.Tensor, HT_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute current between two specific orbitals.
        
        Args:
            psi_total: Total wavefunction [N_time, n]
            HT_pred: Total Hamiltonian [N_time, n, n]
            
        Returns:
            Current tensor [N_time]
        """
        psi_i_conj = tf.math.conj(psi_total[:, self.orbital_i])
        H_ij = HT_pred[:, self.orbital_i, self.orbital_j]
        psi_j = psi_total[:, self.orbital_j]
        
        current = psi_i_conj * H_ij * psi_j
        return current
    
    def get_info(self) -> Dict:
        """Get information about this current calculation."""
        return {
            "type": "orbital_pair",
            "orbital_i": self.orbital_i,
            "orbital_j": self.orbital_j,
            "label": self.label,
            "formula": f"I = ψ_{self.orbital_i}* × H_{self.orbital_i}{self.orbital_j} × ψ_{self.orbital_j}",
            "description": f"Current from orbital {self.orbital_i} to orbital {self.orbital_j}"
        }


class SequentialOrbitalCurrent(CurrentCalculator):
    """
    Current through sequential orbital pairs.
    Useful for current through a chain of orbitals.
    I = Σ ψ_i* × H_i(i+1) × ψ_i+1
    
    User MUST specify both start_orbital and n_pairs. No defaults.
    """
    
    def __init__(self, system, start_orbital: int, n_pairs: int, label: str = ""):
        """
        Initialize sequential orbital current calculator.
        
        Args:
            system: QuantumSystem instance
            start_orbital: Starting orbital index (required)
            n_pairs: Number of consecutive orbital pairs (required)
            label: Optional label for the current
            
        Raises:
            ValueError: If parameters are not provided or invalid
        """
        super().__init__(system)
        
        # Explicitly require both parameters
        if start_orbital is None:
            raise ValueError("start_orbital must be specified in config")
        if n_pairs is None:
            raise ValueError("n_pairs must be specified in config")
        
        self.start_orbital = start_orbital
        self.n_pairs = n_pairs
        self.label = label or f"I_seq_{start_orbital}_{start_orbital+n_pairs}"
        
        # Validate orbital pairs
        self.orbital_pairs = []
        for i in range(n_pairs):
            j = start_orbital + i
            if j + 1 >= system.n:
                raise ValueError(
                    f"Orbital index {j+1} exceeds system dimension {system.n}. "
                    f"With start_orbital={start_orbital} and n_pairs={n_pairs}, "
                    f"the last pair is ({j}, {j+1})"
                )
            self.orbital_pairs.append((j, j + 1))
    
    def compute_current(self, psi_total: tf.Tensor, HT_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute current through sequential orbital pairs.
        
        Args:
            psi_total: Total wavefunction [N_time, n]
            HT_pred: Total Hamiltonian [N_time, n, n]
            
        Returns:
            Current tensor [N_time]
        """
        total_current = tf.zeros_like(psi_total[:, 0], dtype=tf.complex64)
        
        for orbital_i, orbital_j in self.orbital_pairs:
            psi_i_conj = tf.math.conj(psi_total[:, orbital_i])
            H_ij = HT_pred[:, orbital_i, orbital_j]
            psi_j = psi_total[:, orbital_j]
            
            current_pair = psi_i_conj * H_ij * psi_j
            total_current += current_pair
        
        return total_current
    
    def get_info(self) -> Dict:
        """Get information about this current calculation."""
        return {
            "type": "sequential_pairs",
            "start_orbital": self.start_orbital,
            "n_pairs": self.n_pairs,
            "orbital_pairs": self.orbital_pairs,
            "label": self.label,
            "formula": f"I = Σ ψ_i* × H_i(i+1) × ψ_i+1 from {self.start_orbital} to {self.start_orbital + self.n_pairs}",
            "description": f"Current through {self.n_pairs} sequential orbital pairs"
        }


class CurrentFactory:
    """
    Factory class to create current calculators based on user preference.
    No defaults - user must specify all parameters in config.
    """
    
    @staticmethod
    def create_calculator(system, method: str, **kwargs) -> CurrentCalculator:
        """
        Create a current calculator.
        
        Args:
            system: QuantumSystem instance
            method: 'pair' for orbital pair, 'sequential' for sequential pairs
            **kwargs: Method-specific parameters (MUST be provided in config)
                For 'pair': orbital_i, orbital_j
                For 'sequential': start_orbital, n_pairs
                
        Returns:
            CurrentCalculator instance
            
        Raises:
            ValueError: If method is unknown or required parameters are missing
        """
        method = method.lower()
        
        if method == 'pair' or method == 'orbital_pair':
            # Check that required parameters are provided
            if 'orbital_i' not in kwargs:
                raise ValueError("'orbital_i' must be specified in config for pair current")
            if 'orbital_j' not in kwargs:
                raise ValueError("'orbital_j' must be specified in config for pair current")
            
            return OrbitalPairCurrent(
                system,
                orbital_i=kwargs['orbital_i'],
                orbital_j=kwargs['orbital_j'],
                label=kwargs.get('label', '')
            )
        
        elif method == 'sequential' or method == 'seq':
            # Check that required parameters are provided
            if 'start_orbital' not in kwargs:
                raise ValueError("'start_orbital' must be specified in config for sequential current")
            if 'n_pairs' not in kwargs:
                raise ValueError("'n_pairs' must be specified in config for sequential current")
            
            return SequentialOrbitalCurrent(
                system,
                start_orbital=kwargs['start_orbital'],
                n_pairs=kwargs['n_pairs'],
                label=kwargs.get('label', '')
            )
        
        else:
            raise ValueError(f"Unknown current calculation method: {method}. Use 'pair' or 'sequential'")


class CurrentIntegrator:
    """
    Integrate current over energy with Fermi-Dirac distribution.
    """
    
    def __init__(self, system):
        """
        Initialize current integrator.
        
        Args:
            system: QuantumSystem instance
        """
        self.system = system
    
    def integrate_current(self, current_per_mode: tf.Tensor, E_norm: float) -> tf.Tensor:
        """
        Integrate current with Fermi-Dirac distribution.
        
        Args:
            current_per_mode: Current for a specific mode [N_time]
            E_norm: Normalized energy
            
        Returns:
            Weighted current
        """
        fermi = self.system.fermi_dirac(
            E_norm, 
            self.system.Ef_norm, 
            self.system.KBT_norm
        )
        fermi_tensor = tf.cast(fermi, tf.complex64)
        
        return fermi_tensor * current_per_mode
    
    def compute_total_current(self, J_current: tf.Tensor, dE_norm: float) -> tf.Tensor:
        """
        Compute total current with prefactors.
        
        J = -2 Im(∫ dE/(2π) f(E) J(E))
        
        Args:
            J_current: Energy-integrated current
            dE_norm: Normalized energy step
            
        Returns:
            Physical current (normalized)
        """
        prefactor = dE_norm / (2.0 * self.system.const.pi)
        J_norm = prefactor * J_current
        return -2.0 * tf.math.imag(J_norm)
