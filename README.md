# Transport_PINNS

**Simulation du transport électronique quantique dépendant du temps dans des jonctions moléculaires, à l'aide de réseaux de neurones informés par la physique (PINNs).**

---

## 🎯 Rôle du projet

Ce projet calcule **le courant électrique transitoire qui traverse une molécule placée entre deux électrodes métalliques** (par exemple `Pt5H2Pt5` : une molécule de H₂ entre deux chaînes de platine) lorsqu'elle est excitée par une **impulsion laser gaussienne** à l'échelle de la femtoseconde.

Au lieu de résoudre l'équation de Schrödinger dépendante du temps avec un intégrateur numérique classique, le code entraîne un **réseau de neurones (PINN)** dont la fonction de perte impose directement l'équation physique. Le réseau apprend ainsi la fonction d'onde ψ(t), à partir de laquelle on calcule le courant I(t).

En résumé, le code répond à la question :

> *« Si j'éclaire cette jonction moléculaire avec une impulsion lumineuse, quel courant traverse la molécule au cours du temps ? »*

---

## 🧭 Vue d'ensemble du pipeline

Le projet se compose de **deux grandes étapes** :

```
 ┌──────────────────────────────────────────────────────────────────────┐
 │ ÉTAPE 1 — Préparation du système (DFT)            [racine du dépôt]  │
 │                                                                      │
 │  Formule chimique ──► BondDatabase ──► GeometryBuilder ──► Hamiltonian│
 │   ex. "Pt5H2Pt5"      (longueurs de    (positions des     Extractor  │
 │                         liaisons)        atomes, ASE)     (GPAW/LCAO)│
 │                                                              │       │
 │                                    hamiltonian_output/ ◄─────┘       │
 │                                    H_scat, S_scat, H_lead, Σ, Γ, E_F │
 └──────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │ ÉTAPE 2 — Dynamique quantique par PINN         [dossier quantum_pinns]│
 │                                                                      │
 │  HamiltonianLoader ──► QuantumSystem ──► PINNSolver ──► Observables  │
 │  (lit H, Σ, Γ)         (normalisation,   (entraîne un    (courant I(t)│
 │                         états            PINN par        intégré sur │
 │                         stationnaires)   énergie/mode)   l'énergie)  │
 │                                                              │       │
 │                                results_pair_i_j/ ◄───────────┘       │
 │                                courant (.dat) + figures (.png)       │
 └──────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 La physique en quelques lignes

1. **Système** : une région de diffusion (molécule + quelques atomes métalliques) couplée à deux électrodes semi-infinies identiques (système symétrique).
2. **Hamiltonien statique** `H₀` et matrice de recouvrement `S` obtenus par DFT (GPAW, mode LCAO, fonctionnelle PBE).
3. **Self-énergie des électrodes** `Σ` (évaluée au niveau de Fermi) et **matrice de couplage** `Γ = i(Σ − Σ†)`.
4. **États stationnaires** pour chaque énergie `E` et chaque mode α de `Γ` :
   `ψ_st = G^R(E) · √v_α · ξ_α`, avec `G^R(E) = (E − H₀ − Σ)⁻¹`.
5. **Perturbation** : impulsion gaussienne modulée appliquée entre les orbitales 0 et 1 :
   `H_p(t) = −h_e1 · cos(ω(t − t_c)) · exp(−(t − t_c)² / 2W₀²)` (+ une deuxième impulsion retardée optionnelle).
6. **Équation résolue par le PINN** pour la partie perturbée `ψ_p(t)` de la fonction d'onde :
   `i ∂ψ_p/∂t = (H₀ + Σ + H_p(t)) ψ_p + H_p(t) e^{−iEt} ψ_st`, avec `ψ_p(0) = 0`.
7. **Courant** : calculé entre deux orbitales choisies (`I ∝ ψ_i* H_ij ψ_j`), pondéré par la distribution de Fermi-Dirac puis intégré sur l'énergie :
   `I(t) = −2 Im[ ∫ dE/2π · f(E) · J(E, t) ]`.

Toutes les grandeurs sont **adimensionnées** (énergie ÷ max|H₀|, temps ÷ ħ/E_scale) pour stabiliser l'entraînement, puis reconverties en unités physiques (secondes, ampères) à la fin.

---

## 📁 Structure du dépôt

```
Transport_PINNS/
├── run.py                          # ▶ Point d'entrée de l'étape 2 (simulation PINN)
│
├── bond_database.py                # Base de longueurs de liaison typiques (Å)
├── geometry_builder.py             # Construit la géométrie linéaire de la molécule (ASE)
├── hamiltonian_extractor.py        # Calcul DFT (GPAW) → H, S, Σ, Γ, niveaux de Fermi
├── hamiltonian_extractor_example.py# ▶ Point d'entrée de l'étape 1 (pipeline complet DFT)
├── geometry_builder_example.py     # Exemple : construire et visualiser une géométrie
├── bond_example.py                 # Exemple : interroger la base de liaisons
├── hamiltoniend-a.py               # Génère un petit système jouet 3×3 (tests rapides)
├── utils.py                        # Fonctions utilitaires (sauvegarde/chargement de matrices)
│
├── quantum_pinns/                  # Paquet principal de simulation PINN
│   ├── config.py                   # ⚙️  TOUS les paramètres de la simulation
│   ├── hamiltonian_loader.py       # Chargement et validation des matrices
│   ├── system.py                   # Système quantique : normalisation, états stationnaires, impulsion
│   ├── model.py                    # Réseau de neurones (Keras) + fonction de perte physique
│   ├── solver.py                   # Entraînement (Adam puis L-BFGS) et prédiction
│   ├── observables.py              # Calcul et intégration du courant
│   └── utils.py                    # Graphiques et sauvegarde des résultats
│
├── hamiltonian_output/             # Matrices produites par l'étape 1 (exemple Pt5H2Pt5 fourni)
├── geometry_output/                # Géométrie et visualisation de la molécule
└── bond_database_output/           # Longueurs de liaison utilisées
```

---

## ⚙️ Installation

Python ≥ 3.10 recommandé.

```bash
git clone https://github.com/delchere/Transport_PINNS.git
cd Transport_PINNS

# Dépendances de l'étape 2 (simulation PINN)
pip install numpy scipy matplotlib tensorflow

# Dépendances supplémentaires de l'étape 1 (calcul DFT)
pip install ase
# GPAW : voir https://gpaw.readthedocs.io/install.html
# (nécessite aussi les jeux de données PAW : `gpaw install-data <dossier>`)
```

> 💡 GPAW n'est nécessaire **que** pour générer de nouveaux Hamiltoniens. Les matrices de l'exemple `Pt5H2Pt5` sont déjà fournies dans `hamiltonian_output/`, vous pouvez donc lancer directement l'étape 2.

---

## 🚀 Utilisation

Toutes les commandes se lancent **depuis la racine du dépôt** (les chemins sont relatifs).

### Étape 1 — Générer les Hamiltoniens d'une nouvelle molécule (optionnel)

Ouvrez `hamiltonian_extractor_example.py` et changez la formule :

```python
MATERIEL = "Pt5H2Pt5"   # ← votre système : électrode + molécule + électrode
...
extractor = HamiltonianExtractor(atoms_scattering=atoms, lead_type="Pt", n_lead_atoms=4)
```

Puis lancez :

```bash
python hamiltonian_extractor_example.py
```

Cela produit :

| Fichier | Contenu |
|---|---|
| `geometry_output/<formule>_positions.txt`, `_info.txt`, `_visualization.png` | Géométrie de la molécule |
| `hamiltonian_output/H_scat.npy` / `.txt` | Hamiltonien de la région de diffusion |
| `hamiltonian_output/S_scat.txt` | Matrice de recouvrement |
| `hamiltonian_output/H_lead.npy`, `S_lead.txt` | Hamiltonien et recouvrement de l'électrode |
| `hamiltonian_output/sigma.npy` | Self-énergie totale des deux électrodes Σ |
| `hamiltonian_output/gamma.npy` | Matrice de couplage Γ |
| `hamiltonian_output/fermi_levels.txt` | Niveaux de Fermi (région de diffusion, électrode) |

> Pour un test rapide sans DFT, `python hamiltoniend-a.py` génère un système jouet à 3 niveaux (`H_scat.npy`, `sigma.npy`, `gamma.npy` dans le dossier courant). Pensez à adapter `base_dir` dans `config.py` si vous l'utilisez.

### Étape 2 — Lancer la simulation PINN

1. Réglez les paramètres dans **`quantum_pinns/config.py`** (c'est le seul fichier à modifier).
2. Lancez :

```bash
python run.py
```

`run.py` tient en une ligne :

```python
results = QuantumPINN(config).run_full_simulation()
```

---

## 🛠️ Paramètres principaux (`quantum_pinns/config.py`)

| Section | Paramètre | Signification | Défaut |
|---|---|---|---|
| `EnergyParams` | `E_min`, `E_max` | Fenêtre d'énergie d'intégration (eV) | −2.5, 2.5 |
| | `Ne` | Nombre de points d'énergie | 3 |
| | `Ef`, `Temp` | Niveau de Fermi (eV) et température (K) | 0.0, 10 |
| `TimeParams` | `t_min`, `t_max` | Fenêtre temporelle simulée (s) | 0 – 100 fs |
| `GaussianPerturbation` | `cent` | Centre de l'impulsion (s) | 15 fs |
| | `he1`, `he2` | Amplitudes des deux impulsions (eV) | 0.5, 0.0 |
| | `W0` | Largeur de l'impulsion (s) | 5 fs |
| | `tau` | Délai de la 2ᵉ impulsion (s) | 6 fs |
| | `molE` | Fréquence porteuse : ħω = 2·molE (eV) | 0.7 |
| `ModelParams` | `num_hidden_layers`, `num_neurons_per_layer` | Architecture du réseau | 4 × 64 |
| | `N_collocation`, `N_initial`, `N_prediction` | Nombre de points d'entraînement / prédiction | 5000, 1000, 5000 |
| | `N_train_adam`, `N_train_lbfgs` | Itérations Adam puis L-BFGS | 90 000, 700 |
| | `lambda_pde`, `lambda_ic` | Poids de la perte physique / condition initiale | 1.0, 1.0 |
| `CurrentParams` | `method` | `"pair"` (entre 2 orbitales) ou `"sequential"` (suite de paires) | `"pair"` |
| | `orbital_i`, `orbital_j` | Orbitales pour la méthode `pair` | 1, 2 |
| | `start_orbital`, `n_pairs` | Orbitales pour la méthode `sequential` | 0, 5 |
| `DataPaths` | `base_dir` | Dossier contenant `H_scat.npy`, `sigma.npy`, `gamma.npy` | `hamiltonian_output` |
| `SimulationConfig` | `seed` | Graine aléatoire (reproductibilité) | 1234 |

> ⏱️ **Coût de calcul** : un réseau est entraîné **pour chaque couple (énergie, mode)**, soit `Ne × n` entraînements (n = nombre d'orbitales). Avec les valeurs par défaut (90 000 itérations Adam), cela peut prendre longtemps : réduisez `N_train_adam` et `Ne` pour un premier essai.

---

## 📊 Résultats produits

Les résultats sont écrits dans `results_pair_<i>_<j>/` (méthode `pair`) ou `results_seq_<start>_<n>/` (méthode `sequential`) :

| Fichier | Contenu |
|---|---|
| `current_data_physical.dat` | Temps (s) et courant (A) — **résultat principal** |
| `current_data_normalized.dat` | Même chose en unités adimensionnées |
| `time_*.dat`, `current_*.dat` | Colonnes séparées |
| `complex_current.dat` | Parties réelle et imaginaire du courant intégré |
| `perturbation.png`, `training_history.png`, `simulation_summary.png` | Impulsion gaussienne, historique d'entraînement, figure de synthèse |
| `simulation_summary.txt` | Résumé de la simulation et temps de calcul |

---

## 🧠 Détails du réseau de neurones

- **Entrée** : le temps normalisé `t` (1 neurone).
- **Sortie** : `2n` neurones → parties réelle `u(t)` et imaginaire `v(t)` de `ψ_p(t)`.
- **Perte** = `λ_pde · L_EDP + λ_ic · L_CI`
  - `L_EDP` : résidu de l'équation de Schrödinger (séparée en parties réelle/imaginaire), évalué sur des points de collocation tirés aléatoirement, avec les dérivées temporelles obtenues par différentiation automatique.
  - `L_CI` : impose `ψ_p(t = 0) = 0`.
- **Optimisation hybride** : Adam (avec décroissance exponentielle du taux d'apprentissage) puis affinage par L-BFGS (SciPy).

---

## ⚠️ Hypothèses et limites actuelles

- Les électrodes gauche et droite sont **identiques** (système symétrique) : `Σ_total = 2 Σ_électrode`.
- `Σ` et `Γ` sont évaluées **au niveau de Fermi uniquement** puis utilisées pour toutes les énergies (approximation de type « large bande »).
- La géométrie construite est **linéaire** (chaîne atomique le long de x).
- La perturbation agit sur l'élément `H[0, 1]` de l'Hamiltonien.
- Si une longueur de liaison est absente de la base, `BondDatabase` la demande à l'utilisateur au clavier (valeur par défaut 1.5 Å).

---

## 👤 Auteur

[delchere](https://github.com/delchere)
