# test_visu.py
from geometry_builder import GeometryBuilder
from bond_database import BondDatabase

bond_db = BondDatabase()
builder = GeometryBuilder(bond_db)
atoms = builder.build_molecule("Au2H2Au2")
