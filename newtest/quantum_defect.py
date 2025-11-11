# quantum_defect.py
def rydberg_energy(Zeff: float, n: float):  # Hartree
    return - (Zeff**2) / (2.0 * n*n)

def alkali_level(Zeff: float, n: int, delta_l: float):
    return rydberg_energy(Zeff, n - delta_l)
