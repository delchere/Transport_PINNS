# Import direct (sans package)
from bond_database import BondDatabase
from geometry_builder import GeometryBuilder
from hamiltonian_extractor import HamiltonianExtractor

# Configuration simple
MATERIEL = "Pt5H2Pt5"

print(f"🚀 Simulation: {MATERIEL}")

# Exécution
bond_db = BondDatabase()
builder = GeometryBuilder(bond_db)
atoms = builder.build_molecule(MATERIEL, visualize=True)

extractor = HamiltonianExtractor(
    atoms_scattering=atoms,
    lead_type="Pt",
    n_lead_atoms=4
)

extractor.calculate_hamiltonians()
