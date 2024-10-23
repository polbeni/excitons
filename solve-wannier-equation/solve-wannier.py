# Pol Benítez Colominas, September-October 2024
# University of Cambridge and Universitat Politècnica de Catalunya

# This code solves the Wannier Equation for a given 2d material monolayer and provide
# the binding energies and excitonic-wave functions

import numpy as np
import matplotlib.pyplot as plt

# functions
def compute_hamiltonian(m_r, k_mesh, k_step, dielec_layer, dielec_out, state, thick_layer, theta_mesh, theta_step):
    """
    It determines the hamiltonian of the Wannier equation

    Inputs:
        m_r -> relative mass of the exciton system
        k_mesh -> array with the mesh of the momentum space
        k_step -> step in the k_mesh (homogeneous mesh assumed)
        dielec_layer -> dielectric constant of the monolayer
        dielec_out -> dielectric constant of the surroundings of the layer
        state -> the quantum number of the excitonic system (L=0: s, L=1: p, ...)
        thick_layer -> thickness of the monolayer
        theta_mesh -> array with the mesh of the theta angle
        theta_step -> step in the theta_mesh (homogeneous mesh assumed)
    """

    hamiltonian = np.zeros((len(k_mesh), len(k_mesh)), dtype=complex)
    hbar = 0.6582 # eV·fs

    for i_index in range(len(k_mesh)):
        for j_index in range(len(k_mesh)):
            if i_index == j_index:
                element = ((hbar * k_mesh[i_index])**2) / (2 * m_r) 
            else:
                pot = potential(k_mesh[i_index], k_mesh[j_index], dielec_layer, dielec_out, state, thick_layer, theta_mesh, theta_step)
                element = -1 * k_step * k_mesh[j_index] * pot
            
            hamiltonian[i_index, j_index] = element

    for i_index in range(len(k_mesh)):
        for j_index in range(len(k_mesh)):
            if i_index > j_index:
                hamiltonian[i_index, j_index] = np.conjugate(hamiltonian[i_index, j_index])

    return hamiltonian


def potential(k_i, k_j, dielec_layer, dielec_out, state, thick_layer, theta_mesh, theta_step):
    """
    It determines the potential elements for a given potential integrating over theta mesh

    Inputs:
        k_i, k_j -> the two elements of the k-mesh
        dielec_layer -> dielectric constant of the monolayer
        dielec_out -> dielectric constant of the surroundings of the layer
        state -> the quantum number of the excitonic system (L=0: s, L=1: p, ...)
        thick_layer -> thickness of the monolayer
        theta_mesh -> array with the mesh of the theta angle
        theta_step -> step in the theta_mesh (homogeneous mesh assumed)
    """

    value = 0

    for theta_element in theta_mesh:
        q_element = np.sqrt((k_i**2) + (k_j**2) - 2 * k_i * k_j * np.cos(theta_element))
        
        rt_pot_calc = rt_potential(q_element, dielec_layer, dielec_out, thick_layer)

        value = value + theta_step * rt_pot_calc * np.exp(1j * state * theta_element)

    value = (1/ (4 * (np.pi**2))) * value 
    
    return value


def rt_potential(q, dielec_layer, dielec_out, thick_layer):
    """
    It determines the Rytova-Keldysh potential as described in: https://doi.org/10.1021/acs.jpclett.0c02661

    Inputs:
        q -> total momentum
        dielec_layer -> dielectric constant of the monolayer
        dielec_out -> dielectric constant of the surroundings of the layer
        thick_layer -> thickness of the monolayer
    """

    electron_charge = 1 # some units
    vacuum_permittivity = 0.05526 # nm/fs

    r_0 = thick_layer * ((dielec_layer) / (2 * dielec_out))

    potential = (electron_charge**2) / (2 * vacuum_permittivity * dielec_out  * abs(q) * (1 + (r_0 * abs(q))))

    return potential


def eigen_hamiltonian(hamiltonian):
    """
    Determines the eigenvalues and eigenvectors of a given hamiltonian

    Inputs:
        hamiltonian -> the hamiltonian we want to diagonalise
    """

    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)

    for i in range(eigenvectors.shape[1]):
        # Find the index of the first non-zero element
        idx = np.argmax(np.abs(eigenvectors[:, i]) > 1e-10)  # Tolerance for zero values
        
        # Normalize the phase so that the first non-zero element is real and positive
        phase = np.angle(eigenvectors[idx, i])
        eigenvectors[:, i] *= np.exp(-1j * phase)  # Remove phase

    
    return eigenvalues, eigenvectors

# results
relative_mass = 0.108 * 5.68568 # fs^2eV/nm^2
dielectric_layer = 6.1
dielectric_outside = 3.32
state = 0
thickness_layer = 0.636

k_min = 0.001
k_max = 5
number_points_k = 400
k_mesh = np.linspace(k_min, k_max, number_points_k)
k_step = (k_max - k_min) / number_points_k

theta_min = 0
theta_max = 2 * np.pi
number_points_theta = 250
theta_mesh = np.linspace(theta_min, theta_max, number_points_theta, endpoint=False)
theta_step = (theta_max - theta_min) / (number_points_theta-1)
our_hamiltonian = compute_hamiltonian(relative_mass, k_mesh, k_step, dielectric_layer, dielectric_outside, state, thickness_layer, theta_mesh, theta_step)
eigenvalues, eigenvectors = eigen_hamiltonian(our_hamiltonian)
print(eigenvalues[:] * 1000) # in meV


plt.figure()
for x in range(3):
    plt.plot(k_mesh, eigenvectors[:, x], label=f'Exciton wavefunction {x + 1}')
plt.legend()
plt.show()