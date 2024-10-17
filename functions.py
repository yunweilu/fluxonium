import scqubits as scq
import numpy as np
from scipy.special import factorial
# import autograd.numpy as np
import matplotlib.pyplot as plt
import scipy as sci

def sort_eigenpairs(eigenvalues, eigenvectors):
    n = eigenvectors.shape[0]
    sorted_indices = []

    for i in range(n):
        max_abs_vals = np.abs(eigenvectors[i, :])
        max_index = np.argmax(max_abs_vals)
        while max_index in sorted_indices:
            max_abs_vals[max_index] = -np.inf
            max_index = np.argmax(max_abs_vals)
        sorted_indices.append(max_index)

    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    return sorted_eigenvalues, sorted_eigenvectors
def annihilation(dim):
    return np.diag(np.sqrt(np.arange(1,dim)),1)
def creation(dim):
    return np.diag(np.sqrt(np.arange(1,dim)),-1)

def SNAIL(phi_ex,beta,N,Ej,Ec,sdim):
    phi_ex = phi_ex*2*np.pi
    def Us_min(phi_ex):
        def U_s(phi): 
            return (-beta*np.cos(phi-phi_ex)-N*np.cos((phi)/N))
        phi_min = sci.optimize.minimize(U_s,0).x
        return phi_min
    
    def phi_minde(ans, phi_ex):
        def phi_minde_vjp(g):
            c2 = beta*np.cos(ans - phi_ex) + 1/N*np.cos(ans/N)
            return g*beta*np.cos(ans - phi_ex)/c2
        return phi_minde_vjp
    
    phi_min = Us_min(phi_ex)
    # potential expansion around minimum
    c2 = beta*np.cos(phi_min - phi_ex) + 1/N*np.cos(phi_min/N)
    c3 = -(N**2-1)/N**2*beta*np.sin(phi_min-phi_ex)
    omega_s = np.sqrt(8*c2*Ej*Ec)
    phi_zpf = np.power(2*Ec/(Ej*c2),1/4)
    g2 = Ej*phi_zpf**2*c2/2
    g3 = Ej * phi_zpf ** 3 * c3 / 3 / 2
    s = annihilation(sdim)
    sd = creation(sdim)
    x2 = np.matmul(s+sd,s+sd)
    Hs =(omega_s * np.matmul(sd,s)- Ej*(beta*sci.linalg.cosm(phi_zpf*(s+sd)+(phi_min-phi_ex)*np.identity(sdim))
        + N*sci.linalg.cosm((phi_zpf*(s+sd)+phi_min*np.identity(sdim))/N))- g2*x2)
    phi_op = phi_zpf * (s+sd)
    charge_op = -1j*(s-sd)/(2*phi_zpf)
    energy0,U = np.linalg.eigh(Hs)
    energy0,U = sort_eigenpairs(energy0, U)
    energy0 = energy0 - energy0[0]
    Ud = U.transpose().conjugate()
    Hs = Ud@Hs@U
    Hs = Hs - Hs[0,0]*np.identity(sdim)
    return Hs,Ud@phi_op@U,Ud@charge_op@U

def SNAILL(phi_ex,beta,M,EJ,EC,sdim,ELc):
    phi_ext = phi_ex*2*np.pi
    def Us_min(phi_ex):
        def U_s(phi): 
            return (-beta*np.cos(phi-phi_ext)-M*np.cos((phi)/M))
        phi_min = sci.optimize.minimize(U_s,0).x
        return phi_min

    
    phi_min = Us_min(phi_ex)

    # Taylor coefficients of the expansion around potential minimum [ref. B7]
    c2 = beta * np.cos(phi_min - phi_ext) + np.cos(phi_min/M)/M
    c3 = (M**2-1)/M**2 * np.sin(phi_min/M)
    c4 = -beta * np.cos(phi_min - phi_ext) - np.cos(phi_min/M)/M**3
    c5 = (1-M**4)/M**4 * np.sin(phi_min/M)

    p = ELc / (c2 * EJ + ELc)

    # renormalized coefficient due to the linear inductor [ref. B9]
    c2_ti = p * c2
    c3_ti = p**3 * c3
    c4_ti = p**4 * (c4 - 3*c3**2/c2*(1-p))
    c5_ti = p**5 *(c5 - 10*c4*c3/c2*(1-p) + 15*c3**3/c2**2*(1-p)**2)

    # zero point fluctuation of the linearized SNAIL
    phi_c = (2*EC / c2_ti / EJ)**0.25

    # analytical approximation of SNAIL nonlinearity and frequency [ref. B8]
    ws = np.sqrt(8*c2_ti*EC*EJ)
    g3 = EJ * phi_c**3 * c3_ti / factorial(3)
    g4 = EJ * phi_c**4 * c4_ti / factorial(4)
    g5 = EJ * phi_c**5 * c5_ti / factorial(5)
    s = annihilation(sdim)
    sd = creation(sdim)
    x2 = np.matmul(s+sd,s+sd)
    x3 = np.matmul(s+sd,x2)
    x4 = np.matmul(s+sd,x3)
    x5 = np.matmul(s+sd,x4)
    x6 = np.matmul(s+sd,x5)
    Hs = ws*sd@s + g3*x3 + g4*x4 + g5*x5 + 1e-10*x6
    
    phi_op = phi_c * (s+sd)
    charge_op = -1j*(s-sd)/(2*phi_c)
    energy0,U = np.linalg.eigh(Hs)
    energy0,U = sort_eigenpairs(energy0, U)
    energy0 = energy0 - energy0[0]
    Ud = U.transpose().conjugate()
    Hs = Ud@Hs@U
    Hs = Hs - Hs[0,0]*np.identity(sdim)
    return Hs,Ud@phi_op@U,Ud@charge_op@U

def SNAILL_paras(phi_ex,beta,M,EJ,EC,sdim,ELc):
    phi_ext = phi_ex*2*np.pi
    def Us_min(phi_ex):
        def U_s(phi): 
            return (-beta*np.cos(phi-phi_ext)-M*np.cos((phi)/M))
        phi_min = sci.optimize.minimize(U_s,0).x
        return phi_min

    
    phi_min = Us_min(phi_ex)

    # Taylor coefficients of the expansion around potential minimum [ref. B7]
    c2 = beta * np.cos(phi_min - phi_ext) + np.cos(phi_min/M)/M
    c3 = (M**2-1)/M**2 * np.sin(phi_min/M)
    c4 = -beta * np.cos(phi_min - phi_ext) - np.cos(phi_min/M)/M**3
    c5 = (1-M**4)/M**4 * np.sin(phi_min/M)

    p = ELc / (c2 * EJ + ELc)

    # renormalized coefficient due to the linear inductor [ref. B9]
    c2_ti = p * c2
    c3_ti = p**3 * c3
    c4_ti = p**4 * (c4 - 3*c3**2/c2*(1-p))
    c5_ti = p**5 *(c5 - 10*c4*c3/c2*(1-p) + 15*c3**3/c2**2*(1-p)**2)

    # zero point fluctuation of the linearized SNAIL
    phi_c = (2*EC / c2_ti / EJ)**0.25

    # analytical approximation of SNAIL nonlinearity and frequency [ref. B8]
    ws = np.sqrt(8*c2_ti*EC*EJ)
    g3 = EJ * phi_c**3 * c3_ti / factorial(3)
    g4 = EJ * phi_c**4 * c4_ti / factorial(4)
    g5 = EJ * phi_c**5 * c5_ti / factorial(5)
    s = annihilation(sdim)
    sd = creation(sdim)
    x2 = np.matmul(s+sd,s+sd)
    x3 = np.matmul(s+sd,x2)
    x4 = np.matmul(s+sd,x3)
    x5 = np.matmul(s+sd,x4)
    x6 = np.matmul(s+sd,x5)
    Hs = ws*sd@s + g3*x3 + g4*x4 + g5*x5 + 1e-10*x6
    
    phi_op = phi_c * (s+sd)
    charge_op = -1j*(s-sd)/(2*phi_c)
    energy0,U = np.linalg.eigh(Hs)
    energy0,U = sort_eigenpairs(energy0, U)
    energy0 = energy0 - energy0[0]
    Ud = U.transpose().conjugate()
    Hs = Ud@Hs@U
    Hs = Hs - Hs[0,0]*np.identity(sdim)
    return ws,g3,phi_c,c2

def SNAIL_paras(phi_ex,beta,N,Ej,Ec,sdim):
    phi_ex = phi_ex*2*np.pi
    def Us_min(phi_ex):
        def U_s(phi): 
            return (-beta*np.cos(phi-phi_ex)-N*np.cos((phi)/N))
        phi_min = sci.optimize.minimize(U_s,0).x
        return phi_min
    
    def phi_minde(ans, phi_ex):
        def phi_minde_vjp(g):
            c2 = beta*np.cos(ans - phi_ex) + 1/N*np.cos(ans/N)
            return g*beta*np.cos(ans - phi_ex)/c2
        return phi_minde_vjp
    
    phi_min = Us_min(phi_ex)
    # potential expansion around minimum
    c2 = beta*np.cos(phi_min - phi_ex) + 1/N*np.cos(phi_min/N)
    c3 = -(N**2-1)/N**2*beta*np.sin(phi_min-phi_ex)
    
    
    omega_s = np.sqrt(8*c2*Ej*Ec)
    phi_zpf = np.power(2*Ec/(Ej*c2),1/4)
    g2 = Ej*phi_zpf**2*c2/2
    g3 = Ej * phi_zpf ** 3 * c3 / 3 / 2
    s = annihilation(sdim)
    sd = creation(sdim)
    x2 = np.matmul(s+sd,s+sd)
    Hs =omega_s * np.matmul(sd,s)
    phi_op = phi_zpf * (s+sd)
    return omega_s,phi_zpf,Hs,phi_op,g3

def fluxonium(flux1,flux2,EL1,EL2,d1,d2):
    fluxonium1 = scq.Fluxonium(
    EJ=4.9,
    EC=1.7,
    EL=EL1,
    cutoff = 210,
    flux = flux1,
    truncated_dim=d1
    )   
    fluxonium2 =  scq.Fluxonium(
    EJ=4.9,
    EC=2,
    EL=EL2,
    cutoff = 210,
    flux = flux2,
    truncated_dim=d2
    )
    
    H1 = fluxonium1.hamiltonian(energy_esys=True) * 2 * np.pi
    H2 = fluxonium2.hamiltonian(energy_esys=True) * 2 * np.pi
    H1 = H1 - np.identity(d1) * H1[0, 0]
    H2 = H2 - np.identity(d2) * H2[0, 0]
    phi1 = fluxonium1.phi_operator(energy_esys=True)
    phi2 = fluxonium2.phi_operator(energy_esys=True)
    return H1,H2,phi1,phi2


import numpy as np

def shuffle_matrices(H0, V):
    """
    Shuffle the diagonal terms of H0 in ascending order and correspondingly shuffle V.
    
    Parameters:
    H0 (numpy.ndarray): Input matrix to have its diagonal sorted.
    V (numpy.ndarray): Input matrix to be shuffled correspondingly.
    
    Returns:
    tuple: (H0_sorted, V_shuffled) - The sorted H0 and shuffled V matrices.
    """
    
    # Ensure inputs are NumPy arrays
    H0 = np.array(H0)
    V = np.array(V)
    
    # Get the diagonal elements of H0
    diag_elements = np.diag(H0)
    
    # Get the indices that would sort the diagonal elements
    sort_indices = np.argsort(diag_elements)
    
    # Create a new H0 with sorted diagonal elements
    H0_sorted = H0.copy()
    np.fill_diagonal(H0_sorted, np.sort(diag_elements))
    
    # Shuffle the columns of V according to the same sorting
    V_shuffled = V[:, sort_indices]
    
    # If V is square, also shuffle its rows
    if V.shape[0] == V.shape[1]:
        V_shuffled = V_shuffled[sort_indices, :]
    
    return H0_sorted, V_shuffled

# Example usage:
# H0_shuffled, V_shuffled = shuffle_matrices(H0, V)
def split_V(V):
    Vd = np.diag(np.diag(V))  # Diagonal part of V
    Vod = V - Vd  # Off-diagonal part of V
    return Vd, Vod

def commutator(A, B):
    return A @ B - B @ A

def nested_commutator(A, B, n):
    if n == 1:
        return commutator(A, B)
    else:
        return commutator(A, nested_commutator(A, B, n-1))
def create_subspace_projector(dim, subspace_indices):
    """
    Create a projector for the given subspace.

    Args:
        dim (int): The total dimension of the Hilbert space.
        subspace_indices (list or np.ndarray): The indices of the subspace.

    Returns:
        np.ndarray: The projector for the given subspace.
    """
    # Create the basis states for the subspace
    basis_states = [np.zeros(dim) for _ in range(len(subspace_indices))]
    for i, idx in enumerate(subspace_indices):
        basis_states[i][idx] = 1

    # Compute the projector
    projector = np.array([state[:, None] @ state[None, :] for state in basis_states])
    P = np.sum(projector, axis=0)
    Q = np.identity(dim) - P
    return P, Q
def swt_subspace(H0, V, subspace_indices):
    """
    Compute the Schrieffer-Wolff transformation on a subspace of the Hamiltonian.

    Args:
        H0 (np.ndarray): The unperturbed Hamiltonian.
        V (np.ndarray): The perturbation Hamiltonian.
        subspace_indices (list or np.ndarray): The indices of the subspace to consider.

    Returns:
        S (list): The Schrieffer-Wolff transformation operators (S1, S2, S3).
        H (list): The transformed Hamiltonian components (H1, H2, H3, H4).
    """
    dim = H0.shape[0]
    subspace_dim = len(subspace_indices)
    P,Q = create_subspace_projector(dim, subspace_indices)
    
    # Extract the subspace of H0 and V
    
    Vd = P@V@P + Q@V@Q
    Vod = P@V@Q + Q@V@P

    # Compute the energy differences in the subspace
    delta = np.subtract.outer(np.diag(H0), np.diag(H0))
    np.fill_diagonal(delta, 1)

    # Compute the Schrieffer-Wolff transformation components
    H1 = Vd
    S1= Vod/ delta
    H2 = 1/2 * commutator(S1, Vod)
    S2 = -commutator(Vd,S1)/delta

    H3 = 1/2*commutator(S2,Vod)
    S3 = (commutator(S2,Vd) + 1/3*nested_commutator(S1,Vod,2))/delta
    H4 = 1/2*commutator(S3,Vod) - 1/24 *nested_commutator(S1,Vod,3)

    return [S1,S2,S3],[H1,H2,H3,H4]

def operantor_trans(S,op):
    S1,S2 = S
    firstorder = commutator(S1,op)
    secondorder = 0.5*op*S1@S1 + 0.5*S1@S1@op - S1@op@S1 + S2@op - op@S2
    return firstorder,secondorder
    