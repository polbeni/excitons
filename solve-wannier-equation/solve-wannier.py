# Pol Benítez Colominas, September-November 2024
# University of Cambridge and Universitat Politècnica de Catalunya

# This code solves the Wannier Equation for a given 2d material monolayer and provide
# the binding energies and excitonic-wave functions

import numpy as np
import matplotlib.pyplot as plt

# functions
def compute_wannier_matrix(m_r, k_mesh, k_step, dielec_layer, dielec_out, state, thick_layer, theta_mesh, theta_step):
    """
    It determines the Wannier matrix of the Wannier equation

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

    w_ij = np.zeros((len(k_mesh), len(k_mesh)), dtype=complex)
    hbar = 0.6582 # eV·fs
    electron_charge = 1 # eV
    vacuum_permittivity = 0.05526 # nm/fs

    for i_index in range(len(k_mesh)):
        for j_index in range(len(k_mesh)):
            V_coul = 0
            for theta_index in range(len(theta_mesh)):
                if (i_index == j_index) and (theta_index == 0):
                    continue
                else:
                    q_element = np.sqrt((k_mesh[i_index]**2) + (k_mesh[j_index]**2) - (2*k_mesh[i_index]*k_mesh[j_index]*np.cos(theta_mesh[theta_index])))
                    V_coul = V_coul + pot_rt(q_element, dielec_layer, dielec_out, thick_layer, state, theta_mesh[theta_index])

            V_coul = V_coul * ((electron_charge**2) / (((2*np.pi)**2) * vacuum_permittivity * dielec_out)) * k_mesh[j_index] * k_step * theta_step

            w_ij[i_index, j_index] = -V_coul

            if i_index == j_index:
                w_ij[i_index, j_index] = w_ij[i_index, j_index] + (((hbar**2) * (k_mesh[i_index]**2)) / (2 * m_r)) + E_gap

    return w_ij



def pot_rt(q_element, dielec_layer, dielec_out, thick_layer, state, theta_i):
    """
    Computes the Rytova-Keldysh potential as described in: https://doi.org/10.1021/acs.jpclett.0c02661

    Inputs:
        q_element -> q element that depends on k_i, k_j, and the theta angle
        dielec_layer -> dielectric constant of the monolayer
        dielec_out -> dielectric constant of the surroundings of the layer
        thick_layer -> thickness of the monolayer
        state -> the quantum number of the excitonic system (L=0: s, L=1: p, ...)
        theta_i -> the theta angle for the correspondent term in the sum
    """

    r_0 = thick_layer * (dielec_out/ dielec_layer)

    if q_element == 0:
        value = 0
    else:
        value = (1 / (q_element * (1 + q_element * r_0))) * np.exp(1j * state * theta_i)

    return value


def diagonalize_matrix(matrix):
    """
    Determines the eigenvalues and eigenvectors of a given hamiltonian

    Inputs:
        matrix -> the matrix we want to diagonalise
    """

    eigenvalues, eigenvectors = np.linalg.eigh(matrix)

    return eigenvalues, eigenvectors

def normalize_eigenvectors(eigenvectors, k_mesh, k_step):
    """
    Normalize the eigenvectors (wavefunctions)

    Inputs:
        eigenvectors -> eigenvectors solution of the Wannier Equation
        k_mesh -> our mesh of k
        k_step -> the step of the k-mesh
    """

    norm_eigenvectors = []

    for i_it in range(len(k_mesh)):
        norm = 0
        for j_it in range(len(k_mesh)):
            norm = norm + ((1 / (2*np.pi))*k_mesh[j_it]*k_step*(np.absolute(eigenvectors[j_it, i_it])**2))
        
        norm_eigen = []
        for j_it in range(len(k_mesh)):
            norm_eigen.append(eigenvectors[j_it, i_it] / norm)

        norm_eigenvectors.append(norm_eigen)

    return norm_eigenvectors

# results
relative_mass = 0.108 * 5.68568 # fs^2eV/nm^2
dielectric_layer = 6.1
dielectric_outside = 3.32
state = 1
thickness_layer = 0.636

E_gap = 0

k_min = 0.01
k_max = 5
number_points_k = 500
k_mesh = np.linspace(k_min, k_max, number_points_k)
k_step = (k_max - k_min) / number_points_k

theta_min = 0
theta_max = 2 * np.pi
number_points_theta = 250
theta_mesh = np.linspace(theta_min, theta_max, number_points_theta, endpoint=False)
theta_step = (theta_mesh[-1] - theta_mesh[0]) / (len(theta_mesh))
wannier_matrix = compute_wannier_matrix(relative_mass, k_mesh, k_step, dielectric_layer, dielectric_outside, state, thickness_layer, theta_mesh, theta_step)
eigenvalues, eigenvectors = diagonalize_matrix(wannier_matrix)

# normalize the eigenvectors
norm_eigenvectors = normalize_eigenvectors(eigenvectors, k_mesh, k_step)

# print the eigenvalues of the low energy states
print('The Exciton Binding Energies are: ')
for level in range(5):
    print(f'    The exciton level {level + 1}s has a binding energy of: {int(eigenvalues[level] * -1000)} meV')

# plot the results
fig, ax = plt.subplots(2, 1, figsize=(4, 5))
ax[0].set_title('Eigenvalues')
ax[0].set_ylabel('Exciton binding energy (meV)')
ax[0].set_xlabel('Excitonic level')

ax[1].set_title('Eigenvectors')
ax[1].set_ylabel('$\left| \\Psi \\right|^{2}$')
ax[1].set_xlabel('Momentum (nm$^{-1}$)')

ax[0].set_xlim(0, 6)

ax[0].plot([1, 2, 3, 4, 5], eigenvalues[0:5] * -1000, marker='o', linestyle='')

ax[1].set_xlim(0, 1)

for x in range(3):
    ax[1].plot(k_mesh, np.abs(norm_eigenvectors[x][:])**2, label=f'{x + 1}s')

ax[1].legend(frameon=False)

before = [1, 2, 3, 4, 5]
after = ['1s', '2s', '3s', '4s', '5s']
ax[0].set_xticks(ticks=before, labels=after)

plt.tight_layout()
plt.savefig('results.pdf')