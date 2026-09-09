"""
Hamiltonian Extraction for Quantum Transport Calculations
"""

import numpy as np
from gpaw import GPAW, Mixer, FermiDirac
from ase import Atoms
from gpaw.lcao.tightbinding import TightBinding
import os

class HamiltonianExtractor:
    """Hamiltonian extractor for quantum transport"""
    
    def __init__(self, atoms_scattering, lead_atoms=None, lead_type=None, n_lead_atoms=None):
        """
        Initializes the Hamiltonian extractor.
        
        Args:
            atoms_scattering: Scattering region atoms
            lead_atoms: Lead atoms (if None, extracted automatically)
            lead_type: Atomic type of the lead (e.g., 'Pt', 'Au', etc.)
            n_lead_atoms: Number of atoms in the lead
        """
        self.atoms_scattering = atoms_scattering
        
        # Use provided lead_atoms if given
        if lead_atoms is not None:
            self.lead_atoms = lead_atoms
            self.n_lead_atoms = len(lead_atoms)
            self.lead_type = lead_atoms[0].symbol
        else:
            # Otherwise, extract automatically
            self.lead_type = lead_type
            self.n_lead_atoms = n_lead_atoms
            self.lead_atoms = self._extract_lead_atoms(atoms_scattering, lead_type, n_lead_atoms)
        
        # DFT calculation parameters
        self.h = 0.3
        self.xc = 'PBE'
        self.basis = 'szp(dzp)'
        self.occupations = FermiDirac(width=0.1)
        self.mixer = Mixer(0.1, 5, weight=100.0)
        self.kpts_scattering = (1, 1, 1)
        self.kpts_lead = (4, 1, 1)
        
        # Results
        self.H_scat = None
        self.S_scat = None
        self.H_lead = None
        self.S_lead = None
        self.sigma = None
        self.gamma = None
        self.Ef_scat = None
        self.Ef_lead = None
        
        print(f"✅ HamiltonianExtractor initialized")
        print(f"   Scattering: {len(atoms_scattering)} atoms ({atoms_scattering.get_chemical_formula()})")
        print(f"   Lead: {len(self.lead_atoms)} atoms ({self.lead_atoms.get_chemical_formula()})")
    
    def set_calculation_parameters(self, h=None, xc=None, basis=None, occupations=None, 
                                   mixer=None, kpts_scattering=None, kpts_lead=None):
        """Allows user to modify calculation parameters"""
        if h is not None:
            self.h = h
        if xc is not None:
            self.xc = xc
        if basis is not None:
            self.basis = basis
        if occupations is not None:
            self.occupations = occupations
        if mixer is not None:
            self.mixer = mixer
        if kpts_scattering is not None:
            self.kpts_scattering = kpts_scattering
        if kpts_lead is not None:
            self.kpts_lead = kpts_lead
        
        print("✅ Calculation parameters updated")
    
    def _extract_lead_atoms(self, atoms_scattering, lead_type, n_lead_atoms):
        """Extracts atoms for the lead"""
        if lead_type is None or n_lead_atoms is None:
            raise ValueError("If lead_atoms is not provided, lead_type and n_lead_atoms must be specified")
        
        lead_indices = []
        for i, atom in enumerate(atoms_scattering):
            if atom.symbol == lead_type and len(lead_indices) < n_lead_atoms:
                lead_indices.append(i)
        
        if len(lead_indices) < n_lead_atoms:
            raise ValueError(f"Not enough {lead_type} atoms in the scattering region")
        
        lead_atoms = atoms_scattering[lead_indices].copy()
        cell = atoms_scattering.cell.copy()
        cell[0, 0] = cell[0, 0] / 3
        lead_atoms.set_cell(cell)
        
        return lead_atoms
    
    def calculate_hamiltonians(self):
        """Calculates all Hamiltonians"""
        print("\n" + "="*60)
        print("HAMILTONIAN CALCULATION")
        print("="*60)
        
        os.makedirs("hamiltonian_output", exist_ok=True)
        
        # 1. Scattering region
        print("\n1. Scattering region...")
        self.H_scat, self.S_scat, self.Ef_scat = self._calculate_for_atoms(
            self.atoms_scattering, "scattering", self.kpts_scattering
        )
        
        # 2. Lead
        print("\n2. Lead...")
        self.H_lead, self.S_lead, self.Ef_lead = self._calculate_for_atoms(
            self.lead_atoms, "lead", self.kpts_lead
        )
        
        # 3. Sigma and Gamma
        print("\n3. Calculating sigma and gamma matrices...")
        self.sigma, self.gamma = self._calculate_sigma_gamma()
        
        # 4. Save all results
        self._save_all_matrices()
        
        return self.H_scat, self.S_scat, self.sigma, self.gamma
    
    def _calculate_for_atoms(self, atoms, name, kpts):
        """Calculates H, S, and Ef"""
        print(f"   Calculating {name} ({len(atoms)} atoms, kpts={kpts})...")
        
        calc = GPAW(
            h=self.h,
            xc=self.xc,
            basis=self.basis,
            occupations=self.occupations,
            kpts=kpts,
            mode='lcao',
            txt=f'gpaw_calc_{name}.txt',
            mixer=self.mixer,
            symmetry={'point_group': False, 'time_reversal': False}
        )
        
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        Ef = calc.get_fermi_level()
        
        print(f"   ✓ Energy: {energy:.6f} eV")
        print(f"   ✓ Fermi level: {Ef:.6f} eV")
        
        S1 = TightBinding(atoms, calc)
        H_NMM, S_NMM = S1.h_and_s()
        H, S = H_NMM[0], S_NMM[0]
        
        print(f"   ✓ H: {H.shape}, S: {S.shape}")
        
        return H, S, Ef
    
    def _calculate_sigma_gamma(self):
        """Calculates sigma and gamma (symmetric system)"""
        from ase.transport.calculators import TransportCalculator
        from ase.transport.selfenergy import LeadSelfEnergy
        
        # Symmetric system: same lead on both sides
        H_left = self.H_lead
        S_left = self.S_lead
        H_right = self.H_lead
        S_right = self.S_lead
        
        calc = TransportCalculator(
            h=self.H_scat, 
            h1=H_left, 
            h2=H_right,
            s=self.S_scat, 
            s1=S_left, 
            s2=S_right,
            align_bf=1,
        )
        
        h_mm = self.H_scat
        s_mm = self.S_scat
        h_ii = self.H_lead[:len(self.H_lead)//2, :len(self.H_lead)//2]
        h_ij = self.H_lead[:len(self.H_lead)//2, len(self.H_lead)//2:2*len(self.H_lead)//2]
        s_ii = self.S_lead[:len(self.S_lead)//2, :len(self.S_lead)//2]
        s_ij = self.S_lead[:len(self.S_lead)//2, len(self.S_lead)//2:2*len(self.S_lead)//2]
        
        nbf = len(h_mm)
        pl = len(h_ii)
        
        h_im = np.zeros((pl, nbf), complex)
        s_im = np.zeros((pl, nbf), complex)
        h_im[:pl, :pl] = h_ij
        s_im[:pl, :pl] = s_ij
        
        eta = 1e-5
        L = LeadSelfEnergy((h_ii, s_ii), (h_ij, s_ij), (h_im, s_im), eta)
        
        Ef = self.Ef_scat
        print(f"   Using Ef: {Ef:.6f} eV")
        
        sigma_single = L.retarded(Ef)
        sigma = 2 * sigma_single
        
        gamma = 1j * (sigma - np.conjugate(sigma.T))
        
        print(f"   ✓ Sigma (single lead): {sigma_single.shape}")
        print(f"   ✓ Sigma (total): {sigma.shape}")
        print(f"   ✓ Gamma: {gamma.shape}")
        
        return sigma, gamma
    
    def _save_all_matrices(self):
        """Saves all matrices"""
        output_dir = "hamiltonian_output"
        
        print(f"\n💾 Saving in: {output_dir}/")
        
        np.save(f'{output_dir}/H_scat.npy', self.H_scat)
        np.savetxt(f'{output_dir}/H_scat.txt', self.H_scat)
        np.savetxt(f'{output_dir}/S_scat.txt', self.S_scat)
        
        np.save(f'{output_dir}/H_lead.npy', self.H_lead)
        np.savetxt(f'{output_dir}/H_lead.txt', self.H_lead)
        np.savetxt(f'{output_dir}/S_lead.txt', self.S_lead)
        
        np.save(f'{output_dir}/sigma.npy', self.sigma)
        np.save(f'{output_dir}/gamma.npy', self.gamma)
        
        # Compatibility
        np.save('H.npy', self.H_scat)
        np.savetxt('H.txt', self.H_scat)
        np.savetxt('S.txt', self.S_scat)
        np.save('sigma.npy', self.sigma)
        np.save('gamma.npy', self.gamma)
        
        fermi_data = np.array([self.Ef_scat, self.Ef_lead])
        np.savetxt(f'{output_dir}/fermi_levels.txt', fermi_data,
                  header='Ef_scattering Ef_lead (eV)')
        
        print(f"✅ Files saved")
    
    def get_fermi_level(self):
        return self.Ef_scat

