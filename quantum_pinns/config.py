"""
Configuration module for Quantum PINNs.
THE USER SHOULD DIRECTLY MODIFY THE DEFAULT VALUES BELOW.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional
import os


@dataclass
class PhysicalConstants:
    """Physical constants in eV, s, K units"""
    e_charge: float = 1.602176634e-19
    hbar: float = 6.582119514e-16
    KB: float = 8.6173303e-5
    pi: float = np.pi


@dataclass
class EnergyParams:
    """Energy-related parameters - MODIFY HERE"""
    E_min: float = -2.5
    E_max: float = 2.5
    Ne: int = 3
    Ef: float = 0.0
    Temp: float = 10
    
    @property
    def dE(self):
        return (self.E_max - self.E_min) / self.Ne


@dataclass
class TimeParams:
    """Time-related parameters - MODIFY HERE"""
    Power: float = 1.0e-15
    t_min: float = 0.0
    t_max: float = 100.0e-15


@dataclass
class GaussianPerturbation:
    """Gaussian pulse perturbation parameters - MODIFY HERE"""
    cent: float = 15.0e-15
    he1: float = 0.500
    he2: float = 0.00
    W0: float = 5.0e-15
    tau: float = 6.0e-15
    molE: float = 0.7
    
    @property
    def Eg(self):
        return 2 * self.molE
    
    @property
    def W1(self):
        return self.Eg / 6.582119514e-16


@dataclass
class ModelParams:
    """Neural network architecture parameters - MODIFY HERE"""
    num_hidden_layers: int = 4
    num_neurons_per_layer: int = 64
    activation: str = 'tanh'
    kernel_initializer: str = 'glorot_normal'
    
    N_collocation: int = 5000
    N_initial: int = 1000
    N_prediction: int = 5000
    
    N_train_adam: int = 90000
    N_train_lbfgs: int = 700
    adam_initial_lr: float = 1e-3
    adam_decay_steps: int = 30000
    adam_decay_rate: float = 0.9
    
    lambda_pde: float = 1.0
    lambda_ic: float = 1.0


@dataclass
class CurrentParams:
    """
    Current calculation parameters - MODIFY HERE
    Choose 'pair' OR 'sequential'
    """
    # Method selection
    method: str = "pair"  # "pair" or "sequential"
    
    # For current between two orbitals (method "pair")
    orbital_i: int = 1
    orbital_j: int = 2
    
    # For sequential current (method "sequential")
    start_orbital: int = 0
    n_pairs: int = 5
    
    # Optional label
    label: str = "donor_acceptor"
    
    def validate(self, system_dim: int):
        if self.method not in ['pair', 'sequential']:
            raise ValueError("ERROR: current.method must be 'pair' or 'sequential'")
        
        if self.method == 'pair':
            if self.orbital_i < 0 or self.orbital_i >= system_dim:
                raise ValueError(f"orbital_i must be between 0 and {system_dim-1}")
            if self.orbital_j < 0 or self.orbital_j >= system_dim:
                raise ValueError(f"orbital_j must be between 0 and {system_dim-1}")
        elif self.method == 'sequential':
            if self.start_orbital + self.n_pairs >= system_dim:
                raise ValueError(f"Last orbital {self.start_orbital + self.n_pairs} exceeds {system_dim-1}")


@dataclass
class DataPaths:
    """File paths for input data - MODIFY HERE"""
    base_dir: str = "hamiltonian_output"
    H_file: str = "H_scat.npy"
    sigma_file: str = "sigma.npy"
    gamma_file: str = "gamma.npy"
    # S_file: str = "S_scat.txt"
    
    @property
    def H_path(self):
        return os.path.join(self.base_dir, self.H_file)
    
    @property
    def sigma_path(self):
        return os.path.join(self.base_dir, self.sigma_file)
    
    @property
    def gamma_path(self):
        return os.path.join(self.base_dir, self.gamma_file)
    
    # @property
    # def S_path(self):
    #     return os.path.join(self.base_dir, self.S_file)


@dataclass
class SimulationConfig:
    """Main configuration container - ALL DEFAULT VALUES ARE SET HERE"""
    # Constants
    physical: PhysicalConstants = field(default_factory=PhysicalConstants)
    
    # Physics parameters
    energy: EnergyParams = field(default_factory=EnergyParams)
    time: TimeParams = field(default_factory=TimeParams)
    perturbation: GaussianPerturbation = field(default_factory=GaussianPerturbation)
    
    # Neural network
    model: ModelParams = field(default_factory=ModelParams)
    
    # Current calculation
    current: CurrentParams = field(default_factory=CurrentParams)
    
    # File paths
    paths: DataPaths = field(default_factory=DataPaths)
    
    # System info
    lead_type: str = "Pt"   #symbol of the atom use in the lead
    n_lead_atoms: int = 4   #Number of atoms use for lead 
    
    # Output
    output_dir: str = "quantum_pinns_results"
    
    # Reproducibility
    seed: int = 1234
    
    def __post_init__(self):
        """Update output directory based on current method"""
        if self.current.method == "pair":
            self.output_dir = f"results_pair_{self.current.orbital_i}_{self.current.orbital_j}"
        else:
            self.output_dir = f"results_seq_{self.current.start_orbital}_{self.current.n_pairs}"


# Single global instance - ALL DEFAULT VALUES ABOVE
config = SimulationConfig()

