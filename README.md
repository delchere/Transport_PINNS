# Transport_PINNS

**Simulation of time-dependent quantum electron transport through molecular junctions using Physics-Informed Neural Networks (PINNs).**

---

## 🎯 What this project does

This project computes **the transient electric current flowing through a molecule placed between two metallic electrodes** (for example `Pt5H2Pt5`: an H₂ molecule between two platinum chains) when the junction is excited by a **femtosecond Gaussian laser pulse**.

Instead of solving the time-dependent Schrödinger equation with a classical numerical integrator, the code trains a **neural network (PINN)** whose loss function directly enforces the physical equation. The network learns the wavefunction ψ(t), from which the current I(t) is computed.

In short, the code answers the question:

> *"If I shine a light pulse on this molecular junction, what current flows through the molecule over time?"*

---

## 🧭 Pipeline overview

The project consists of **two main stages**:

```
 ┌──────────────────────────────────────────────────────────────────────┐
 │ STAGE 1 — System preparation (DFT)                 [repository root] │
 │                                                                      │
 │  Chemical formula ──► BondDatabase ──► GeometryBuilder ──► Hamiltonian│
 │   e.g. "Pt5H2Pt5"     (bond lengths)   (atomic positions,  Extractor │
 │                                          ASE)              (GPAW/LCAO)│
 │                                                              │       │
 │                                    hamiltonian_output/ ◄─────┘       │
 │                                    H_scat, S_scat, H_lead, Σ, Γ, E_F │
 └──────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │ STAGE 2 — Quantum dynamics with PINNs        [quantum_pinns package] │
 │                                                                      │
 │  HamiltonianLoader ──► QuantumSystem ──► PINNSolver ──► Observables  │
 │  (reads H, Σ, Γ)       (normalization,   (trains one     (current I(t)│
 │                         stationary       PINN per        integrated  │
 │                         states)          energy/mode)    over energy)│
 │                                                              │       │
 │                                results_pair_i_j/ ◄───────────┘       │
 │                                current (.dat) + figures (.png)       │
 └──────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 The physics in a nutshell

1. **System**: a scattering region (molecule + a few metal atoms) coupled to two identical semi-infinite electrodes (symmetric system).
2. **Static Hamiltonian** `H₀` and overlap matrix `S` obtained from DFT (GPAW, LCAO mode, PBE functional).
3. **Lead self-energy** `Σ` (evaluated at the Fermi level) and **coupling matrix** `Γ = i(Σ − Σ†)`.
4. **Stationary states** for each energy `E` and each eigenmode α of `Γ`:
   `ψ_st = G^R(E) · √v_α · ξ_α`, with `G^R(E) = (E − H₀ − Σ)⁻¹`.
5. **Perturbation**: a modulated Gaussian pulse applied between orbitals 0 and 1:
   `H_p(t) = −h_e1 · cos(ω(t − t_c)) · exp(−(t − t_c)² / 2W₀²)` (+ an optional second, delayed pulse).
6. **Equation solved by the PINN** for the perturbed part `ψ_p(t)` of the wavefunction:
   `i ∂ψ_p/∂t = (H₀ + Σ + H_p(t)) ψ_p + H_p(t) e^{−iEt} ψ_st`, with `ψ_p(0) = 0`.
7. **Current**: computed between two chosen orbitals (`I ∝ ψ_i* H_ij ψ_j`), weighted by the Fermi–Dirac distribution and integrated over energy:
   `I(t) = −2 Im[ ∫ dE/2π · f(E) · J(E, t) ]`.

All quantities are **non-dimensionalized** (energy ÷ max|H₀|, time ÷ ħ/E_scale) to stabilize training, then converted back to physical units (seconds, amperes) at the end.

---

## 📁 Repository structure

```
Transport_PINNS/
├── run.py                          # ▶ Entry point for stage 2 (PINN simulation)
│
├── bond_database.py                # Database of typical bond lengths (Å)
├── geometry_builder.py             # Builds the linear geometry of the molecule (ASE)
├── hamiltonian_extractor.py        # DFT calculation (GPAW) → H, S, Σ, Γ, Fermi levels
├── hamiltonian_extractor_example.py# ▶ Entry point for stage 1 (full DFT pipeline)
├── geometry_builder_example.py     # Example: build and visualize a geometry
├── bond_example.py                 # Example: query the bond database
├── hamiltoniend-a.py               # Generates a small 3×3 toy system (quick tests)
├── utils.py                        # Utility functions (save/load matrices)
│
├── quantum_pinns/                  # Main PINN simulation package
│   ├── config.py                   # ⚙️  ALL simulation parameters
│   ├── hamiltonian_loader.py       # Loads and validates the matrices
│   ├── system.py                   # Quantum system: normalization, stationary states, pulse
│   ├── model.py                    # Neural network (Keras) + physics-informed loss
│   ├── solver.py                   # Training (Adam then L-BFGS) and prediction
│   ├── observables.py              # Current calculation and integration
│   └── utils.py                    # Plotting and result saving
│
├── hamiltonian_output/             # Matrices produced by stage 1 (Pt5H2Pt5 example included)
├── geometry_output/                # Molecule geometry and visualization
└── bond_database_output/           # Bond lengths used
```

---

## ⚙️ Installation

Python ≥ 3.10 recommended.

```bash
git clone https://github.com/delchere/Transport_PINNS.git
cd Transport_PINNS

# Stage 2 dependencies (PINN simulation)
pip install numpy scipy matplotlib tensorflow

# Additional stage 1 dependencies (DFT calculation)
pip install ase
# GPAW: see https://gpaw.readthedocs.io/install.html
# (also requires the PAW datasets: `gpaw install-data <directory>`)
```

> 💡 GPAW is **only** needed to generate new Hamiltonians. The matrices for the `Pt5H2Pt5` example are already provided in `hamiltonian_output/`, so you can run stage 2 directly.

---

## 🚀 Usage

All commands must be run **from the repository root** (paths are relative).

### Stage 1 — Generate the Hamiltonians for a new molecule (optional)

Open `hamiltonian_extractor_example.py` and change the formula:

```python
MATERIEL = "Pt5H2Pt5"   # ← your system: electrode + molecule + electrode
...
extractor = HamiltonianExtractor(atoms_scattering=atoms, lead_type="Pt", n_lead_atoms=4)
```

Then run:

```bash
python hamiltonian_extractor_example.py
```

This produces:

| File | Content |
|---|---|
| `geometry_output/<formula>_positions.txt`, `_info.txt`, `_visualization.png` | Molecule geometry |
| `hamiltonian_output/H_scat.npy` / `.txt` | Hamiltonian of the scattering region |
| `hamiltonian_output/S_scat.txt` | Overlap matrix |
| `hamiltonian_output/H_lead.npy`, `S_lead.txt` | Lead Hamiltonian and overlap |
| `hamiltonian_output/sigma.npy` | Total self-energy of both leads Σ |
| `hamiltonian_output/gamma.npy` | Coupling matrix Γ |
| `hamiltonian_output/fermi_levels.txt` | Fermi levels (scattering region, lead) |

> For a quick test without DFT, `python hamiltoniend-a.py` generates a 3-level toy system (`H_scat.npy`, `sigma.npy`, `gamma.npy` in the current directory). Remember to adjust `base_dir` in `config.py` if you use it.

### Stage 2 — Run the PINN simulation

1. Set the parameters in **`quantum_pinns/config.py`** (the only file you need to edit).
2. Run:

```bash
python run.py
```

`run.py` is a one-liner:

```python
results = QuantumPINN(config).run_full_simulation()
```

---

## 🛠️ Main parameters (`quantum_pinns/config.py`)

| Section | Parameter | Meaning | Default |
|---|---|---|---|
| `EnergyParams` | `E_min`, `E_max` | Energy integration window (eV) | −2.5, 2.5 |
| | `Ne` | Number of energy points | 3 |
| | `Ef`, `Temp` | Fermi level (eV) and temperature (K) | 0.0, 10 |
| `TimeParams` | `t_min`, `t_max` | Simulated time window (s) | 0 – 100 fs |
| `GaussianPerturbation` | `cent` | Pulse center (s) | 15 fs |
| | `he1`, `he2` | Amplitudes of the two pulses (eV) | 0.5, 0.0 |
| | `W0` | Pulse width (s) | 5 fs |
| | `tau` | Delay of the 2nd pulse (s) | 6 fs |
| | `molE` | Carrier frequency: ħω = 2·molE (eV) | 0.7 |
| `ModelParams` | `num_hidden_layers`, `num_neurons_per_layer` | Network architecture | 4 × 64 |
| | `N_collocation`, `N_initial`, `N_prediction` | Number of training / prediction points | 5000, 1000, 5000 |
| | `N_train_adam`, `N_train_lbfgs` | Adam then L-BFGS iterations | 90,000, 700 |
| | `lambda_pde`, `lambda_ic` | Weights of the physics / initial-condition losses | 1.0, 1.0 |
| `CurrentParams` | `method` | `"pair"` (between 2 orbitals) or `"sequential"` (chain of pairs) | `"pair"` |
| | `orbital_i`, `orbital_j` | Orbitals for the `pair` method | 1, 2 |
| | `start_orbital`, `n_pairs` | Orbitals for the `sequential` method | 0, 5 |
| `DataPaths` | `base_dir` | Folder containing `H_scat.npy`, `sigma.npy`, `gamma.npy` | `hamiltonian_output` |
| `SimulationConfig` | `seed` | Random seed (reproducibility) | 1234 |

> ⏱️ **Computational cost**: one network is trained **for each (energy, mode) pair**, i.e. `Ne × n` trainings (n = number of orbitals). With the default values (90,000 Adam iterations) this can take a long time: reduce `N_train_adam` and `Ne` for a first try.

---

## 📊 Output

Results are written to `results_pair_<i>_<j>/` (`pair` method) or `results_seq_<start>_<n>/` (`sequential` method):

| File | Content |
|---|---|
| `current_data_physical.dat` | Time (s) and current (A) — **main result** |
| `current_data_normalized.dat` | Same, in dimensionless units |
| `time_*.dat`, `current_*.dat` | Separate columns |
| `complex_current.dat` | Real and imaginary parts of the integrated current |
| `perturbation.png`, `training_history.png`, `simulation_summary.png` | Gaussian pulse, training history, summary figure |
| `simulation_summary.txt` | Simulation summary and run time |

---

## 🧠 Neural network details

- **Input**: normalized time `t` (1 neuron).
- **Output**: `2n` neurons → real part `u(t)` and imaginary part `v(t)` of `ψ_p(t)`.
- **Loss** = `λ_pde · L_PDE + λ_ic · L_IC`
  - `L_PDE`: residual of the Schrödinger equation (split into real and imaginary parts), evaluated on randomly sampled collocation points, with time derivatives obtained by automatic differentiation.
  - `L_IC`: enforces `ψ_p(t = 0) = 0`.
- **Hybrid optimization**: Adam (with exponential learning-rate decay) followed by L-BFGS refinement (SciPy).

---

## ⚠️ Current assumptions and limitations

- The left and right electrodes are **identical** (symmetric system): `Σ_total = 2 Σ_lead`.
- `Σ` and `Γ` are evaluated **at the Fermi level only** and then used for all energies (wide-band-like approximation).
- The generated geometry is **linear** (atomic chain along x).
- The perturbation acts on the `H[0, 1]` element of the Hamiltonian.
- If a bond length is missing from the database, `BondDatabase` asks the user for it on the keyboard (default 1.5 Å).

---

## 👤 Author

[delchere](https://github.com/delchere)
