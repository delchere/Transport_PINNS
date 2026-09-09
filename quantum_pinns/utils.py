"""
Utilities module.
Provides visualization, data saving, and helper functions.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime


class Visualizer:
    """
    Visualization utilities for quantum PINN results.
    """
    
    def __init__(self, system, output_dir: str = "quantum_pinns_results"):
        """
        Initialize visualizer.
        
        Args:
            system: QuantumSystem instance
            output_dir: Directory to save figures
        """
        self.system = system
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_current(self, time_values: np.ndarray, current_values: np.ndarray,
                    title: str = "Current", label: str = "Current",
                    save: bool = True, filename: str = "current.png",
                    physical_units: bool = True, ax=None):
        """
        Plot current vs time.
        
        Args:
            time_values: Time array
            current_values: Current array
            title: Plot title
            label: Legend label
            save: Whether to save the figure
            filename: Output filename
            physical_units: If True, use physical units (fs, A)
            ax: Matplotlib axis (if None, creates new figure)
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        if physical_units:
            time_plot = time_values * 1e15  # Convert to fs
            ylabel = "Current (A)"
            time_label = "Time (fs)"
        else:
            time_plot = time_values
            ylabel = "Current (normalized)"
            time_label = "Time (normalized)"
        
        ax.plot(time_plot, current_values, label=label, linewidth=2)
        ax.set_xlabel(time_label, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        if save:
            plt.savefig(os.path.join(self.output_dir, filename), 
                       dpi=300, bbox_inches='tight')
    
    def plot_current_components(self, time_values: np.ndarray, 
                               complex_current: np.ndarray,
                               save: bool = True, 
                               filename: str = "current_components.png"):
        """
        Plot real and imaginary parts of complex current.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(time_values, np.real(complex_current), 
                label='Re(J)', color='red', linewidth=1.5, alpha=0.8)
        ax.plot(time_values, np.imag(complex_current), 
                label='Im(J)', color='green', linewidth=1.5, alpha=0.8)
        
        ax.set_xlabel("Time (normalized)", fontsize=12)
        ax.set_ylabel("Complex Current", fontsize=12)
        ax.set_title("Current Components", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save:
            plt.savefig(os.path.join(self.output_dir, filename), 
                       dpi=300, bbox_inches='tight')
    
    def plot_perturbation(self, save: bool = True, 
                         filename: str = "perturbation.png"):
        """
        Plot Gaussian perturbation.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        t_pert = np.linspace(self.system.t_min_norm, self.system.t_max_norm, 1000)
        pert_values = []
        
        for t in t_pert:
            H_pert = self.system.perturb_gauss(t)
            pert_values.append(np.real(H_pert[0, 1]))
        
        ax.plot(t_pert, pert_values, label='H₀₁(t)', 
                color='darkgreen', linewidth=2)
        ax.set_xlabel("Time (normalized)", fontsize=12)
        ax.set_ylabel("Perturbation amplitude (normalized)", fontsize=12)
        ax.set_title("Gaussian Pulse Perturbation", fontsize=14)
        ax.grid(True, alpha=0.3)
        
        if save:
            plt.savefig(os.path.join(self.output_dir, filename), 
                       dpi=300, bbox_inches='tight')
    
    def plot_training_history(self, history: Dict, save: bool = True,
                             filename: str = "training_history.png"):
        """
        Plot Adam training history.
        """
        if 'adam_history' not in history:
            print("No Adam history found")
            return
        
        adam_hist = history['adam_history']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.semilogy(adam_hist['total'], label='Total Loss', 
                   color='blue', linewidth=1)
        ax.semilogy(adam_hist['pde'], label='PDE Loss', 
                   color='orange', linewidth=1)
        ax.semilogy(adam_hist['ic'], label='IC Loss', 
                   color='purple', linewidth=1)
        
        ax.set_xlabel("Epochs", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title("Adam Training Convergence", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save:
            plt.savefig(os.path.join(self.output_dir, filename), 
                       dpi=300, bbox_inches='tight')
    
    def create_summary_figure(self, results: Dict, histories: List = None):
        """
        Create comprehensive summary figure with multiple panels.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Panel 1: Normalized current
        ax1 = axes[0, 0]
        self.plot_current(
            results['time_normalized'], 
            results['current_normalized'],
            title="Normalized Current",
            label="CJJ(t)",
            save=False,
            physical_units=False,
            ax=ax1
        )
        
        # Panel 2: Current components
        ax2 = axes[0, 1]
        ax2.plot(results['time_normalized'], np.real(results['complex_current']), 
                label='Re(J)', color='red', linewidth=1, alpha=0.8)
        ax2.plot(results['time_normalized'], np.imag(results['complex_current']), 
                label='Im(J)', color='green', linewidth=1, alpha=0.8)
        ax2.set_xlabel("Time (normalized)", fontsize=12)
        ax2.set_ylabel("Complex Current", fontsize=12)
        ax2.set_title("Current Components", fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: Training history (if available)
        ax3 = axes[1, 0]
        if histories and len(histories) > 0 and 'adam_history' in histories[0]:
            adam_hist = histories[0]['adam_history']
            ax3.semilogy(adam_hist['total'], label='Total', 
                        color='blue', linewidth=1)
            ax3.semilogy(adam_hist['pde'], label='PDE', 
                        color='orange', linewidth=1)
            ax3.semilogy(adam_hist['ic'], label='IC', 
                        color='purple', linewidth=1)
            ax3.set_xlabel("Epochs", fontsize=12)
            ax3.set_ylabel("Loss", fontsize=12)
            ax3.set_title("Adam Convergence", fontsize=14)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Panel 4: Physical current
        ax4 = axes[1, 1]
        self.plot_current(
            results['time_physical'],
            results['current_physical'],
            title="Physical Current",
            label="I(t)",
            save=False,
            physical_units=True,
            ax=ax4
        )
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "simulation_summary.png"), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig


class DataSaver:
    """
    Save simulation results to files.
    """
    
    def __init__(self, output_dir: str = "quantum_pinns_results"):
        """
        Initialize data saver.
        
        Args:
            output_dir: Directory to save data
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def save_current(self, t_norm: np.ndarray, t_phys: np.ndarray,
                    J_norm: np.ndarray, CJJ_norm: np.ndarray, 
                    CJJ_phys: np.ndarray):
        """
        Save current data to files.
        """
        # Normalized data
        np.savetxt(
            os.path.join(self.output_dir, "time_normalized.dat"),
            t_norm,
            header="Time_normalized",
            fmt='%.10e'
        )
        np.savetxt(
            os.path.join(self.output_dir, "current_normalized.dat"),
            CJJ_norm,
            header="Current_normalized",
            fmt='%.10e'
        )
        
        # Physical data
        np.savetxt(
            os.path.join(self.output_dir, "time_physical.dat"),
            t_phys,
            header="Time_seconds",
            fmt='%.10e'
        )
        np.savetxt(
            os.path.join(self.output_dir, "current_physical.dat"),
            CJJ_phys,
            header="Current_amperes",
            fmt='%.10e'
        )
        
        # Combined data files
        np.savetxt(
            os.path.join(self.output_dir, "current_data_normalized.dat"),
            np.column_stack((t_norm, CJJ_norm)),
            header="Time_normalized Current_normalized",
            fmt='%.10e'
        )
        np.savetxt(
            os.path.join(self.output_dir, "current_data_physical.dat"),
            np.column_stack((t_phys, CJJ_phys)),
            header="Time_seconds Current_amperes",
            fmt='%.10e'
        )
        
        # Complex current
        np.savetxt(
            os.path.join(self.output_dir, "complex_current.dat"),
            np.column_stack((t_norm, np.real(J_norm), np.imag(J_norm))),
            header="Time_normalized Re(J) Im(J)",
            fmt='%.10e'
        )
        
        print(f"✅ Current data saved to {self.output_dir}")
    
    def save_config(self, config):
        """Save configuration to JSON file."""
        config_path = os.path.join(self.output_dir, "config.json")
        config.save(config_path)
        print(f"✅ Configuration saved to {config_path}")
    
    def save_summary(self, results: Dict, simulation_time: float):
        """
        Save simulation summary to text file.
        """
        summary_path = os.path.join(self.output_dir, "simulation_summary.txt")
        
        with open(summary_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("QUANTUM PINN SIMULATION SUMMARY\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total simulation time: {simulation_time:.2f} seconds ({simulation_time/60:.2f} minutes)\n\n")
            
            f.write("Current statistics:\n")
            f.write(f"  Max normalized current: {np.max(np.abs(results['current_normalized'])):.6e}\n")
            f.write(f"  Max physical current: {np.max(np.abs(results['current_physical'])):.6e} A\n")
            f.write(f"  Final normalized current: {results['current_normalized'][-1]:.6e}\n")
            f.write(f"  Final physical current: {results['current_physical'][-1]:.6e} A\n")
            f.write(f"  Current scale factor: {results['current_scale']:.6e} A\n\n")
            
            if 'histories' in results and results['histories']:
                final_losses = [h.get('final_loss', 0) for h in results['histories'] if 'final_loss' in h]
                if final_losses:
                    f.write("Training statistics:\n")
                    f.write(f"  Average final loss: {np.mean(final_losses):.6e}\n")
                    f.write(f"  Max final loss: {np.max(final_losses):.6e}\n")
                    f.write(f"  Min final loss: {np.min(final_losses):.6e}\n")
        
        print(f"✅ Summary saved to {summary_path}")


def setup_tensorflow(seed: int = 1234):
    """Setup TensorFlow with deterministic behavior."""
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
    # Limit GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU memory growth enabled for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"⚠️ GPU configuration error: {e}")
    
    print(f"✅ TensorFlow seeded with {seed}")


def time_function(func):
    """Decorator to time functions."""
    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        end = time()
        print(f"⏱️ {func.__name__} took {end - start:.2f} seconds")
        return result
    return wrapper
