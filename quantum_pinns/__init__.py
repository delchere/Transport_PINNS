"""
Quantum PINNs Package
Quantum simulation using Physics-Informed Neural Networks
"""

from quantum_pinns.config import config, SimulationConfig
from quantum_pinns.hamiltonian_loader import HamiltonianLoader
from quantum_pinns.system import QuantumSystem
from quantum_pinns.model import PINNModel, LossComputer  
from quantum_pinns.solver import PINNSolver
from quantum_pinns.observables import (
    CurrentFactory, 
    CurrentIntegrator, 
    OrbitalPairCurrent, 
    SequentialOrbitalCurrent
)
from quantum_pinns.utils import Visualizer, DataSaver, setup_tensorflow, time_function

# Alias for convenience
QuantumPINN = PINNModel  

# Package version
__version__ = "1.0.0"

# Public API
__all__ = [
    'config',
    'SimulationConfig',
    'QuantumSystem',
    'HamiltonianLoader',
    'PINNModel',     
    'LossComputer',
    'PINNSolver',
    'CurrentFactory',
    'CurrentIntegrator',
    'OrbitalPairCurrent',
    'SequentialOrbitalCurrent',
    'Visualizer',
    'DataSaver',
    'setup_tensorflow',
    'time_function',
    'QuantumPINN'     
]

