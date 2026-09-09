"""
Solver module.
Handles the hybrid training (Adam + L-BFGS) of PINN models.
"""

import tensorflow as tf
import numpy as np
import scipy.optimize
import gc
from time import time
from typing import Dict, Tuple, Optional, List, Callable
import os


class PINNSolver:
    """
    Solver for quantum PINN with hybrid Adam + L-BFGS training.
    """
    
    def __init__(self, config, system, model_builder, loss_computer):
        """
        Initialize solver.
        
        Args:
            config: SimulationConfig
            system: QuantumSystem instance
            model_builder: PINNModel instance
            loss_computer: LossComputer instance
        """
        self.config = config
        self.system = system
        self.model_builder = model_builder
        self.loss_computer = loss_computer
        self.model_params = config.model
        
        # Storage for training history
        self.history = {}
    
    def solve_for_energy_mode(self, i_energy: int, j_mode: int):
        """
        Solve for a specific energy and mode.
        
        Args:
            i_energy: Energy index
            j_mode: Mode index
            
        Returns:
            Dictionary with results
        """
        n = self.system.n
        
        # Energy
        E_norm = self.system.E_min_norm + i_energy * self.system.dE_norm
        
        # Stationary state
        psi_st = self.system.get_stationary_state(i_energy, j_mode)
        psi_st_tensor = tf.convert_to_tensor(psi_st, dtype=tf.complex64)
        
        # Time grids - Utilisation de random.uniform comme dans le code original
        t_colloc_norm = tf.random.uniform(
            (self.model_params.N_collocation, 1),
            minval=self.system.t_min_norm,
            maxval=self.system.t_max_norm,
            dtype=tf.float32
        )
        t_0_norm = tf.zeros((self.model_params.N_initial, 1), dtype=tf.float32) + self.system.t_min_norm
        
        # Perturbation Hamiltonians
        HP_list = []
        for k in range(self.model_params.N_collocation):
            t_k_norm = float(t_colloc_norm[k, 0])
            HP_k = self.system.perturb_gauss(t_k_norm)
            HP_list.append(HP_k)
        
        HP_tensor = tf.convert_to_tensor(np.stack(HP_list), dtype=tf.complex64)
        
        # Total Hamiltonian
        H_0_tensor = tf.cast(self.system.H_0_norm, tf.complex64)
        SELF_tensor = tf.cast(self.system.SELF_norm, tf.complex64)
        
        H_0_expanded = tf.repeat(tf.expand_dims(H_0_tensor, axis=0), self.model_params.N_collocation, axis=0)
        SELF_expanded = tf.repeat(tf.expand_dims(SELF_tensor, axis=0), self.model_params.N_collocation, axis=0)
        
        H_t = HP_tensor + H_0_expanded + SELF_expanded
        H_t_r = tf.math.real(H_t)
        H_t_i = tf.math.imag(H_t)
        
        # Build model if not already built
        if self.model_builder.model is None:
            self.model_builder.build(n)
        
        model_instance = self.model_builder.get_model()
        
        # Train
        history = self.train_hybrid(
            model_instance, t_colloc_norm, t_0_norm, E_norm,
            psi_st_tensor, HP_tensor, H_t_r, H_t_i
        )
        
        # Clean up
        del t_colloc_norm, t_0_norm, HP_tensor, H_t_r, H_t_i
        tf.keras.backend.clear_session()
        gc.collect()
        
        return {
            'model': model_instance,
            'history': history,
            'E_norm': E_norm,
            'psi_st_tensor': psi_st_tensor,
            'i_energy': i_energy,
            'j_mode': j_mode
        }
    
    def train_hybrid(self, model: tf.keras.Model, t_colloc_norm: tf.Tensor,
                    t_0_norm: tf.Tensor, E_norm: float, psi_st_tensor: tf.Tensor,
                    HP_tensor: tf.Tensor, H_t_r: tf.Tensor, H_t_i: tf.Tensor) -> Dict:
        """
        Train model with Adam then L-BFGS.
        """
        # Phase 1: Entraînement avec Adam
        print("Phase 1: Entraînement avec Adam...")
        
        # Learning rate schedule
        initial_lr = self.model_params.adam_initial_lr
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=self.model_params.adam_decay_steps,
            decay_rate=self.model_params.adam_decay_rate,
            staircase=True
        )
        optimizer_adam = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        
        @tf.function
        def adam_train_step():
            with tf.GradientTape() as tape:
                total, pde, ic = self.loss_computer.total_loss(
                    model, t_colloc_norm, t_0_norm, E_norm, 
                    psi_st_tensor, HP_tensor, H_t_r, H_t_i
                )
            gradients = tape.gradient(total, model.trainable_variables)
            optimizer_adam.apply_gradients(zip(gradients, model.trainable_variables))
            return total, pde, ic
        
        adam_history = {'total': [], 'pde': [], 'ic': []}
        t_start_adam = time()
        
        for epoch in range(self.model_params.N_train_adam):
            loss, pde_loss, ic_loss = adam_train_step()
            
            if epoch % 1000 == 0:
                print(f"Adam Epoch {epoch:5d}: Total = {loss:.6e}, PDE = {pde_loss:.6e}, IC = {ic_loss:.6e}")
            
            adam_history['total'].append(float(loss))
            adam_history['pde'].append(float(pde_loss))
            adam_history['ic'].append(float(ic_loss))
            
            # Early stopping
            if epoch > 1000 and loss < 1e-4:
                print(f"Arrêt précoce d'Adam à l'époque {epoch}")
                break
        
        print(f"Temps d'entraînement Adam: {time() - t_start_adam:.2f} secondes")
        print(f"Perte finale Adam: {loss:.6e}")
        
        del optimizer_adam
        tf.keras.backend.clear_session()
        gc.collect()
        
        # Phase 2: Raffinement avec L-BFGS
        print("\nPhase 2: Raffinement avec L-BFGS...")
        
        t_start_lbfgs = time()
        
        # Poids initiaux
        initial_weights = self.model_builder.get_weights_vector()
        
        # Configuration L-BFGS
        lbfgs_options = {
            'maxiter': self.model_params.N_train_lbfgs,
            'maxfun': self.model_params.N_train_lbfgs * 2,
            'ftol': 1e-12,
            'gtol': 1e-12,
            'eps': 1e-8,
            'disp': True
        }
        
        # Variables pour le suivi
        self.lbfgs_iter = 0
        self.lbfgs_losses = []
        
        def lbfgs_loss_func(weights_1d):
            """Fonction de perte pour L-BFGS."""
            self.model_builder.set_weights_vector(weights_1d)
            
            with tf.GradientTape() as tape:
                loss, pde_loss, ic_loss = self.loss_computer.total_loss(
                    model, t_colloc_norm, t_0_norm, E_norm,
                    psi_st_tensor, HP_tensor, H_t_r, H_t_i
                )
            
            gradients = tape.gradient(loss, model.trainable_variables)
            grad_1d = []
            for grad in gradients:
                if grad is not None:
                    grad_1d.append(grad.numpy().flatten())
            grad_1d = np.concatenate(grad_1d) if grad_1d else np.array([])
            
            return float(loss.numpy()), grad_1d.astype(np.float64)
        
        def callback_func(xk):
            self.lbfgs_iter += 1
            if self.lbfgs_iter % 100 == 0:
                self.model_builder.set_weights_vector(xk)
                loss, pde_loss, ic_loss = self.loss_computer.total_loss(
                    model, t_colloc_norm, t_0_norm, E_norm,
                    psi_st_tensor, HP_tensor, H_t_r, H_t_i
                )
                print(f"L-BFGS Iter {self.lbfgs_iter:4d}: Total = {loss:.6e}, PDE = {pde_loss:.6e}, IC = {ic_loss:.6e}")
                self.lbfgs_losses.append(float(loss))
        
        # Optimisation L-BFGS
        try:
            result = scipy.optimize.minimize(
                fun=lbfgs_loss_func,
                x0=initial_weights.astype(np.float64),
                method='L-BFGS-B',
                jac=True,
                callback=callback_func,
                options=lbfgs_options
            )
            
            self.model_builder.set_weights_vector(result.x)
            print(f"L-BFGS terminé: {result.message}")
            print(f"Nombre d'itérations: {result.nit}")
            print(f"Perte finale: {result.fun:.6e}")
            
        except Exception as e:
            print(f"Erreur L-BFGS: {e}")
            print("Utilisation des poids d'Adam comme solution finale")
        
        print(f"Temps d'entraînement L-BFGS: {time() - t_start_lbfgs:.2f} secondes")
        
        # Évaluation finale
        final_loss, final_pde_loss, final_ic_loss = self.loss_computer.total_loss(
            model, t_colloc_norm, t_0_norm, E_norm,
            psi_st_tensor, HP_tensor, H_t_r, H_t_i
        )
        print(f"Perte finale combinée: Total = {final_loss:.6e}, PDE = {final_pde_loss:.6e}, IC = {final_ic_loss:.6e}")
        
        return {
            'adam_history': adam_history,
            'lbfgs_losses': self.lbfgs_losses,
            'final_loss': float(final_loss),
            'final_pde_loss': float(final_pde_loss),
            'final_ic_loss': float(final_ic_loss)
        }
    
    def predict(self, model: tf.keras.Model, E_norm: float, 
               psi_st_tensor: tf.Tensor, N_pred: Optional[int] = None) -> Dict:
        """
        Make predictions with trained model.
        """
        if N_pred is None:
            N_pred = self.model_params.N_prediction
        
        # Time grid for prediction
        t_pred_norm_vals = np.linspace(
            self.system.t_min_norm, 
            self.system.t_max_norm, 
            N_pred
        )
        t_pred_norm = tf.convert_to_tensor(t_pred_norm_vals.reshape(-1, 1), dtype=tf.float32)
        
        # Perturbation Hamiltonians for prediction
        HP_pred_list = []
        for s in range(N_pred):
            t_s_norm = t_pred_norm_vals[s]
            HP_s = self.system.perturb_gauss(t_s_norm)
            HP_pred_list.append(HP_s)
        
        HP_pred = tf.convert_to_tensor(np.stack(HP_pred_list), dtype=tf.complex64)
        
        # Model prediction
        psi_pred = model(t_pred_norm)
        u_pred = psi_pred[:, :self.system.n]
        v_pred = psi_pred[:, self.system.n:]
        psi_p = tf.complex(u_pred, v_pred)
        
        # Add stationary state with phase
        arg = -E_norm * t_pred_norm_vals
        phase = tf.exp(1j * tf.cast(arg, tf.complex64))
        phase = tf.expand_dims(phase, axis=-1)
        
        psi_st_broadcast = tf.transpose(psi_st_tensor)
        psi_total = psi_p + phase * psi_st_broadcast
        
        # Total Hamiltonian for prediction
        H_0_tensor = tf.cast(self.system.H_0_norm, tf.complex64)
        SELF_tensor = tf.cast(self.system.SELF_norm, tf.complex64)
        
        H_0_pred = tf.repeat(tf.expand_dims(H_0_tensor, axis=0), N_pred, axis=0)
        SELF_pred = tf.repeat(tf.expand_dims(SELF_tensor, axis=0), N_pred, axis=0)
        
        HT_pred = HP_pred + H_0_pred + SELF_pred
        
        return {
            'psi_total': psi_total,
            'psi_p': psi_p,
            'HT_pred': HT_pred,
            't_pred_norm': t_pred_norm_vals,
            'phase': phase
        }
    
    def run_full_simulation(self, current_calculator, current_integrator):
        """
        Exécute la simulation complète pour toutes les énergies et tous les modes.
        
        Args:
            current_calculator: CurrentCalculator instance
            current_integrator: CurrentIntegrator instance
            
        Returns:
            Dictionary avec tous les résultats
        """
        print("\n" + "="*60)
        print("🚀 SIMULATION QUANTUM PINN COMPLÈTE")
        print("="*60)
        print(f"   Dimension système: {self.system.n}")
        print(f"   Énergies: {self.config.energy.Ne}")
        print(f"   Modes par énergie: {self.system.n}")
        print(f"   Total simulations: {self.config.energy.Ne * self.system.n}")
        print(f"   Méthode courant: {current_calculator.get_info()['type']}")
        print("="*60)
        
        N_pred = self.model_params.N_prediction
        J_current = tf.zeros((N_pred,), dtype=tf.complex64)
        all_histories = []
        
        t_start = time()
        
        # Boucle principale sur les énergies
        for i in range(self.config.energy.Ne):
            print(f"\n📊 Énergie {i+1}/{self.config.energy.Ne}")
            
            E_norm = self.system.E_min_norm + i * self.system.dE_norm
            print(f"   E = {E_norm * self.system.E_scale:.3f} eV")
            
            current_mode_sum = tf.zeros((N_pred,), dtype=tf.complex64)
            
            for j in range(self.system.n):
                print(f"   🔄 Mode {j+1}/{self.system.n}", end="\r")
                
                # Résolution pour ce mode et cette énergie
                result = self.solve_for_energy_mode(i, j)
                all_histories.append(result['history'])
                
                # Prédiction
                prediction = self.predict(
                    result['model'],
                    result['E_norm'],
                    result['psi_st_tensor'],
                    N_pred
                )
                
                # Calcul du courant
                current_mode = current_calculator.compute_current(
                    prediction['psi_total'],
                    prediction['HT_pred']
                )
                
                # Intégration avec Fermi-Dirac
                weighted_current = current_integrator.integrate_current(
                    current_mode, result['E_norm']
                )
                current_mode_sum += weighted_current
                
                # Nettoyage mémoire
                del result['model']
                tf.keras.backend.clear_session()
                gc.collect()
            
            J_current += current_mode_sum
            print(f"   ✅ Énergie {i+1} ajoutée                                  ")
        
        # Calcul final du courant
        CJJ_norm = current_integrator.compute_total_current(
            J_current, self.system.dE_norm
        ).numpy()
        
        CJJ_phys = CJJ_norm * self.system.current_scale
        
        t_pred_norm = np.linspace(
            self.system.t_min_norm, self.system.t_max_norm, N_pred
        )
        t_phys = t_pred_norm * self.system.t_scale
        
        total_time = time() - t_start
        
        print(f"\n" + "="*60)
        print(f"✅ SIMULATION TERMINÉE")
        print(f"   Temps total: {total_time:.2f} s ({total_time/60:.2f} min)")
        print(f"   Courant max: {np.max(np.abs(CJJ_phys)):.3e} A")
        print("="*60)
        
        return {
            'current_normalized': CJJ_norm,
            'current_physical': CJJ_phys,
            'time_normalized': t_pred_norm,
            'time_physical': t_phys,
            'complex_current': J_current.numpy(),
            'current_scale': self.system.current_scale,
            'histories': all_histories,
            'total_time': total_time
        }
