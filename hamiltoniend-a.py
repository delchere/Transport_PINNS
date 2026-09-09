
import numpy as np

# H_scat.npy : 0.7 sur diagonale sauf -0.7, et 0.1, 0.1
H_scat = np.array([
    [-0.7, 0.0, 0.0],
    [0.0, 0.7, 0.1],
    [0.0, 0.1, 0.7]
], dtype=complex)

# sigma.npy : -0.05j sur toute la diagonale
sigma = np.array([
    [-0.05j, 0.0, 0.0],
    [0.0, -0.05j, 0.0],
    [0.0, 0.0, -0.05j]
], dtype=complex)

# gamma.npy : 0.05 sur toute la diagonale
gamma = np.array([
    [0.05, 0.0, 0.0],
    [0.0, 0.05, 0.0],
    [0.0, 0.0, 0.05]
], dtype=complex)

# Sauvegarde
np.save('H_scat.npy', H_scat)
np.save('sigma.npy', sigma)
np.save('gamma.npy', gamma)

print("Fichiers créés :")
print("H_scat.npy :")
print(H_scat)
print("\nsigma.npy :")
print(sigma)
print("\ngamma.npy :")
print(gamma)
