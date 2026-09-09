"""
Exemple simple d'utilisation - L'utilisateur met juste sa formule
"""

from bond_database import BondDatabase

# 1. Créer la base de données
db = BondDatabase()

# 2. Mettre la formule
liaisons = db.get_required_bonds("Au3H2Au3")  # ← Change "C6H6" pour ta formule

# 3. C'est tout! Le module gère l'affichage et la sauvegarde
