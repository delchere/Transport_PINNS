#!/usr/bin/env python3
"""
Quantum PINN
"""

from quantum_pinns import QuantumPINN  
from quantum_pinns.config import config

# SINGLE LINE TO RUN THE FULL SIMULATION
results = QuantumPINN(config).run_full_simulation()

