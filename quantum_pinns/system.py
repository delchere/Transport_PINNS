"""
Quantum System module.
Handles physical parameters, normalization, and stationary states.
"""

import numpy as np
import tensorflow as tf
from typing import Dict, Tuple, Optional, List
import scipy.linalg


class QuantumSystem:
    """
    Quantum transport system with normalization and stationary states.
    """
    
    def __init__(self, config, hamiltonian_data):
        """
        Initialize quantum system.
        
        Args:
            config: SimulationConfig object
            hamiltonian_data: Dictionary with Hamiltonian matrices
        """
        self.config = config
        self.data = hamiltonian_data
        
        # Physical constants
        self.const = config.physical
        
        # Extract matrices
        self.H_0 = hamiltonian_data['H_0']
        self.SELF = hamiltonian_data['SELF']
        self.Gamma = hamiltonian_data['Gamma']
        self.S = hamiltonian_data.get('S', None)
        
        # System dimension
        self.n = self.H_0.shape[0]
        
        # Setup normalization scales
        self._setup_normalization()
        
        # Normalize all matrices
        self._normalize_matrices()
        
        # Store stationary states
        self.stock_psi = None
    
    def _setup_normalization(self):
        """Configure normalization scales based on physical scales."""
        # Energy scale: absolute maximum of H_0
        self.E_scale = np.max(np.abs(self.H_0))
        
        # Time scale: ℏ / E_scale
        self.t_scale = self.const.hbar / self.E_scale
        
        # Wavefunction scale: dimensionless
        self.psi_scale = 1.0
        
        # Current scale: e / t_scale (Amperes)
        self.current_scale = self.const.e_charge / self.t_scale
        
        print("\n" + "="*60)
        print("NORMALIZATION SCALES")
        print("="*60)
        print(f"  Energy scale      : {self.E_scale:.3f} eV")
        print(f"  Time scale        : {self.t_scale:.2e} s")
        print(f"  Current scale     : {self.current_scale:.6e} A")
        print(f"  ℏ/t_scale         : {self.const.hbar/self.t_scale:.3f} eV = E_scale ✓")
        print("="*60)
    
    def _normalize_matrices(self):
        """Normalize all matrices by energy scale."""
        self.H_0_norm = self.H_0 / self.E_scale
        self.SELF_norm = self.SELF / self.E_scale
        self.Gamma_norm = self.Gamma / self.E_scale
        
        if self.S is not None:
            self.S_norm = self.S / self.E_scale
        
        # Normalize energy parameters
        self.E_min_norm = self.config.energy.E_min / self.E_scale
        self.E_max_norm = self.config.energy.E_max / self.E_scale
        self.dE_norm = self.config.energy.dE / self.E_scale
        self.Ef_norm = self.config.energy.Ef / self.E_scale
        self.KBT_norm = (self.config.energy.Temp * self.const.KB) / self.E_scale
        
        # Normalize time parameters
        self.t_min_norm = self.config.time.t_min / self.t_scale
        self.t_max_norm = self.config.time.t_max / self.t_scale
        
        # Normalize perturbation parameters
        self.cent_norm = self.config.perturbation.cent / self.t_scale
        self.he1_norm = self.config.perturbation.he1 / self.E_scale
        self.he2_norm = self.config.perturbation.he2 / self.E_scale
        self.W0_norm = self.config.perturbation.W0 / self.t_scale
        self.W1_norm = (self.config.perturbation.molE * 2 / self.const.hbar) * self.t_scale
        self.tau_norm = self.config.perturbation.tau / self.t_scale
        
        print(f"✅ Matrices normalized by E_scale = {self.E_scale:.3f} eV")
        print(f"✅ Time parameters normalized by t_scale = {self.t_scale:.2e} s")
    
    def fermi_dirac(self, E_norm: float, mu_norm: float, KBT_norm: float) -> float:
        """Fermi-Dirac distribution with normalized parameters."""
        if KBT_norm == 0:
            return 1.0 if E_norm <= mu_norm else 0.0
        return 1.0 / (1.0 + np.exp((E_norm - mu_norm) / KBT_norm))
    
    def solve_stationary_states(self):
        """
        Solve stationary states for all energy points.
        
        Returns:
            Array of stationary states with shape (Ne, n, n)
        """
        Ne = self.config.energy.Ne
        n = self.n
        
        self.stock_psi = np.zeros((Ne, n, n), dtype=complex)
        
        for i in range(Ne):
            E_norm = self.E_min_norm + i * self.dE_norm
            
            # Retarded Green's function
            a = E_norm * np.eye(n, dtype=complex)
            GR = np.linalg.inv(a - self.H_0_norm - self.SELF_norm)
            
            # Diagonalize Gamma
            eigvals, eigvecs = np.linalg.eigh(self.Gamma_norm)
            
            for j in range(n):
                v_alpha = eigvals[j]
                xi_alpha = eigvecs[:, j:j+1]
                rhs = np.lib.scimath.sqrt(v_alpha) * xi_alpha
                psi_st = GR @ rhs
                self.stock_psi[i, j, :] = psi_st[:, 0]
        
        print(f"✅ Stationary states computed: {Ne} energies × {n} modes")
        
        return self.stock_psi
    
    def perturb_gauss(self, t_norm: float) -> np.ndarray:
        """
        Gaussian pulse perturbation.
        
        Args:
            t_norm: Normalized time
            
        Returns:
            Perturbation matrix H_p(t)
        """
        n = self.n
        How = np.zeros((n, n), dtype=complex)
        
        value = (-self.he1_norm * np.cos(self.W1_norm * (t_norm - self.cent_norm)) *
                 np.exp(-((t_norm - self.cent_norm)**2) / (2 * self.W0_norm**2)) -
                 self.he2_norm * np.cos(self.W1_norm * (t_norm - self.cent_norm - self.tau_norm)) *
                 np.exp(-((t_norm - self.cent_norm - self.tau_norm)**2) / (2 * self.W0_norm**2)))
    
        How[0, 1] = value
        How[1, 0] = value
        
        return How
    
    def get_stationary_state(self, i_energy: int, j_mode: int) -> np.ndarray:
        """
        Get stationary state for specific energy and mode.
        
        Args:
            i_energy: Energy index
            j_mode: Mode index
            
        Returns:
            Stationary state vector
        """
        if self.stock_psi is None:
            self.solve_stationary_states()
        
        return self.stock_psi[i_energy, j_mode, :].reshape((self.n, 1))
    
    def get_total_hamiltonian(self, t_norm: float) -> np.ndarray:
        """
        Get total Hamiltonian H(t) = H_0 + Σ + H_p(t).
        
        Args:
            t_norm: Normalized time
            
        Returns:
            Total Hamiltonian matrix
        """
        H_p = self.perturb_gauss(t_norm)
        return H_p + self.H_0_norm + self.SELF_norm
    
    def get_normalization_info(self) -> Dict:
        """Return normalization information as dictionary."""
        return {
            'E_scale': self.E_scale,
            't_scale': self.t_scale,
            'current_scale': self.current_scale,
            'E_min_norm': self.E_min_norm,
            'E_max_norm': self.E_max_norm,
            't_min_norm': self.t_min_norm,
            't_max_norm': self.t_max_norm
        }
