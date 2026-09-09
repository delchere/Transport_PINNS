"""
PINN Model module.
Defines the neural network architecture for the Physics-Informed Neural Network (PINN).
"""

import tensorflow as tf
import numpy as np
from typing import Tuple, Dict, Optional, List

# Imports for the run_full_simulation method
from quantum_pinns.hamiltonian_loader import HamiltonianLoader
from quantum_pinns.system import QuantumSystem
from quantum_pinns.solver import PINNSolver
from quantum_pinns.observables import CurrentFactory, CurrentIntegrator
from quantum_pinns.utils import Visualizer, DataSaver, setup_tensorflow
# from quantum_pinns.model import LossComputer


class PINNModel:
    """
    Physics-Informed Neural Network for quantum wavefunctions.
    Predicts real and imaginary parts of the wavefunction.
    """
    
    def __init__(self, config):
        """
        Initialize PINN model.
        
        Args:
            config: SimulationConfig object with model parameters
        """
        self.config = config.model
        self.system_dim = None  # Will be set when building the model
        self.model = None
        self.full_config = config  # Keep full config for run_full_simulation
    
    def build(self, system_dim: int) -> tf.keras.Model:
        """
        Build the neural network architecture.
        
        Args:
            system_dim: Dimension of the Hilbert space (number of orbitals)
            
        Returns:
            Compiled Keras model
        """
        self.system_dim = system_dim
        model = tf.keras.Sequential(name="QuantumPINN")
        
        # Input layer (time)
        model.add(tf.keras.layers.InputLayer(input_shape=(1,), name="input_time"))
        
        # Hidden layers
        for i in range(self.config.num_hidden_layers):
            model.add(tf.keras.layers.Dense(
                self.config.num_neurons_per_layer,
                activation=self.config.activation,
                kernel_initializer=self.config.kernel_initializer,
                name=f"hidden_{i+1}"
            ))
        
        # Output layer: 2 * n neurons (real + imaginary parts)
        model.add(tf.keras.layers.Dense(
            2 * system_dim,
            activation='linear',
            name="output_wavefunction"
        ))
        
        self.model = model
        
        print(f"\n✅ PINN model built:")
        print(f"   Input: time (1)")
        print(f"   Hidden layers: {self.config.num_hidden_layers} × {self.config.num_neurons_per_layer} neurons")
        print(f"   Output: {2 * system_dim} (ψ_real, ψ_imag)")
        print(f"   Total parameters: {model.count_params():,}")
        
        return model
    
    def get_model(self) -> tf.keras.Model:
        """Return the built model."""
        if self.model is None:
            raise ValueError("Model not built yet. Call build() first.")
        return self.model
    
    def predict_wavefunction(self, t_norm: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Predict wavefunction at given times.
        
        Args:
            t_norm: Normalized time tensor [batch, 1]
            
        Returns:
            Tuple of (real_part, imag_part), each [batch, n]
        """
        psi_pred = self.model(t_norm)
        u = psi_pred[:, :self.system_dim]  # Real part
        v = psi_pred[:, self.system_dim:]  # Imaginary part
        return u, v
    
    def get_weights_vector(self) -> np.ndarray:
        """Flatten model weights into a 1D vector for L-BFGS."""
        weights = []
        for layer in self.model.layers:
            if hasattr(layer, 'get_weights') and layer.get_weights():
                for w in layer.get_weights():
                    weights.append(w.flatten())
        return np.concatenate(weights) if weights else np.array([])
    
    def set_weights_vector(self, weights_1d: np.ndarray):
        """Set model weights from a 1D vector."""
        idx = 0
        for layer in self.model.layers:
            if hasattr(layer, 'get_weights') and layer.get_weights():
                layer_weights = []
                for w in layer.get_weights():
                    size = w.size
                    new_w = weights_1d[idx:idx+size].reshape(w.shape)
                    layer_weights.append(new_w)
                    idx += size
                layer.set_weights(layer_weights)
    
    def summary(self):
        """Print model summary."""
        if self.model:
            self.model.summary()
        else:
            print("Model not built yet.")
    
    def run_full_simulation(self, config=None):
        """
        Simple interface to run the full simulation.
        Called from run.py in a single line.
        
        Args:
            config: Optional SimulationConfig object (uses self.full_config if None)
            
        Returns:
            Dictionary containing all results
        """
        if config is None:
            config = self.full_config
        
        # TensorFlow setup
        setup_tensorflow(config.seed)
        
        # Load Hamiltonian
        print("\n📂 Loading Hamiltonian...")
        loader = HamiltonianLoader(config)
        hamiltonian_data = loader.load_all()
        
        # Initialize quantum system
        print("\n⚛️ Initializing quantum system...")
        system = QuantumSystem(config, hamiltonian_data)
        system.solve_stationary_states()
        
        # Build model if not already built
        if self.model is None:
            self.build(system.n)
        
        # Loss computer
        loss_computer = LossComputer(config, system)
        
        # Solver
        solver = PINNSolver(config, system, self, loss_computer)
        
        # Current calculation
        print("\n🔌 Setting up current calculation...")
        current_calc = CurrentFactory.create_calculator(
            system,
            method=config.current.method,
            orbital_i=config.current.orbital_i,
            orbital_j=config.current.orbital_j,
            start_orbital=config.current.start_orbital,
            n_pairs=config.current.n_pairs,
            label=config.current.label
        )
        
        # Integrator
        current_integrator = CurrentIntegrator(system)
        
        # Run the simulation
        print("\n🚀 Running full simulation...")
        results = solver.run_full_simulation(current_calc, current_integrator)
        
        # Save results
        print("\n💾 Saving results...")
        saver = DataSaver(config.output_dir)
        saver.save_current(
            results['time_normalized'],
            results['time_physical'],
            results['complex_current'],
            results['current_normalized'],
            results['current_physical']
        )
        
        # Save summary
        saver.save_summary(results, results['total_time'])
        
        # Visualization
        print("\n📊 Generating plots...")
        visualizer = Visualizer(system, config.output_dir)
        
        # Gaussian pulse plot
        print("   - Gaussian perturbation...")
        visualizer.plot_perturbation(save=True)
        
        # Training history (first mode)
        if results['histories'] and len(results['histories']) > 0:
            print("   - Training history (first mode)...")
            visualizer.plot_training_history(results['histories'][0], save=True)
        
        # Main summary figure
        visualizer.create_summary_figure(results, results['histories'])
        
        print(f"\n✅ Simulation completed. Results in: {config.output_dir}")
        
        return results


class LossComputer:
    """
    Computes physics-informed and initial condition losses for the PINN.
    """
    
    def __init__(self, config, system):
        """
        Initialize the loss computer.
        
        Args:
            config: SimulationConfig object
            system: QuantumSystem instance
        """
        self.config = config
        self.system = system
        self.model_params = config.model
    
    def physics_loss(self, model: tf.keras.Model, t_colloc_norm: tf.Tensor, 
                    E_norm: float, psi_st_tensor: tf.Tensor, 
                    HP_tensor: tf.Tensor, H_t_r: tf.Tensor, H_t_i: tf.Tensor) -> tf.Tensor:
        """
        Compute physics-informed PDE loss.
        iℏ ∂ψ/∂t = H(t)ψ + S(t)
        Split into real and imaginary parts.
        """
        n = self.system.n
        
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(t_colloc_norm)
            psi_pred = model(t_colloc_norm)
            u = psi_pred[:, :n]  # Real part
            v = psi_pred[:, n:]  # Imag part
        
        # Time derivatives
        u_t = tape.batch_jacobian(u, t_colloc_norm)[:, :, 0]
        v_t = tape.batch_jacobian(v, t_colloc_norm)[:, :, 0]
        del tape
        
        # Source term S(t) = -iℏ e^{-iEt/ℏ} H_p ψ_st
        E_complex = tf.cast(E_norm, tf.complex64)
        t_complex = tf.cast(tf.squeeze(t_colloc_norm), tf.complex64)
        phase = tf.exp(-1j * E_complex * t_complex)
        phase = tf.expand_dims(phase, axis=-1)
        
        psi_st_broadcast = tf.transpose(psi_st_tensor)
        expo_term = phase * psi_st_broadcast
        expo_term = tf.expand_dims(expo_term, axis=-1)
        
        S_complex = tf.matmul(HP_tensor, expo_term)
        S_complex = tf.squeeze(S_complex, axis=-1)
        S_r = tf.math.real(S_complex)
        S_i = tf.math.imag(S_complex)
        
        # Residual equations
        u_exp = tf.expand_dims(u, axis=-1)
        v_exp = tf.expand_dims(v, axis=-1)
        
        # ∂u/∂t = H_r v + H_i u - S_i
        r_u = u_t - tf.squeeze(tf.matmul(H_t_r, v_exp), axis=-1) - \
              tf.squeeze(tf.matmul(H_t_i, u_exp), axis=-1) - S_i
        
        # -∂v/∂t = H_r u - H_i v + S_r
        r_v = -v_t - tf.squeeze(tf.matmul(H_t_r, u_exp), axis=-1) + \
              tf.squeeze(tf.matmul(H_t_i, v_exp), axis=-1) - S_r
        
        pde_loss = tf.reduce_mean(tf.square(r_u)) + tf.reduce_mean(tf.square(r_v))
        return pde_loss
    
    def initial_condition_loss(self, model: tf.keras.Model, t_0_norm: tf.Tensor) -> tf.Tensor:
        """Compute initial condition loss (ψ(t=0) = 0)."""
        psi_pred_0 = model(t_0_norm)
        n = self.system.n
        u_0 = psi_pred_0[:, :n]
        v_0 = psi_pred_0[:, n:]
        ic_loss = tf.reduce_mean(tf.square(u_0)) + tf.reduce_mean(tf.square(v_0))
        return ic_loss
    
    def total_loss(self, model: tf.keras.Model, t_colloc_norm: tf.Tensor,
                  t_0_norm: tf.Tensor, E_norm: float, psi_st_tensor: tf.Tensor,
                  HP_tensor: tf.Tensor, H_t_r: tf.Tensor, H_t_i: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Compute total weighted loss."""
        pde_loss = self.physics_loss(model, t_colloc_norm, E_norm, psi_st_tensor, HP_tensor, H_t_r, H_t_i)
        ic_loss = self.initial_condition_loss(model, t_0_norm)
        
        total = (self.model_params.lambda_pde * pde_loss + 
                 self.model_params.lambda_ic * ic_loss)
        
        return total, pde_loss, ic_loss

