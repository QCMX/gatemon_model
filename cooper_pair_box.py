"""
Numerically solve the Hamiltonian of the Cooper Pair Box (CPB).
"""

import numpy as np


def Mathieu_solutions(z, nu, eta, N):
    r"""Find Mathieu functions $\psi(z)$ with Floquet exponent $\nu$.
    Will find the first M=2*N+1 solutions, sorted ascending by
    characteristic value a.

    $$\partial_z^2 \psi + (a - 2\eta \cos(2z)) \psi = 0$$

    Large $\nu$ results in less accurate results. Use the periodicity
    in $\nu$ to stay at $\nu$ close to zero.

    Eigenfunctions are normalized to <psi|psi> = 2pi with
    integration of z from 0 to 2pi.

    See also R. B. Shirts, "The computation of eigenvalues and solutions of Mathieu's differential equation for noninteger order",
    ACM Transactions on Mathematical Software 19, 377-390 (1993), doi:10.1145/155743.155796

    Parameters
    ----------
    z : 1-D numpy.ndarray
        Points to evaluate solution on
    nu : float
        Floquet exponent
    eta : float
        Parameter of the Mathieu equation
    N : int
        Size of matrix used to solve for coefficients.
        Matrix used has size (M,M), M=2*N+1.

    Returns
    -------
    a : numpy.ndarray of shape (M,)
        Characteristic values a.
    c : numpy.ndarray of shape (M, M)
        Solution coefficients. Column i corresponding to i-th characteristic value.
    psi : numpy.ndarray of shape (len(z),M)
        Solutions evaluated at z (along first axis).
        Column i (second axis) corresponding to i-th characteristic value.
    """
    # This would also need A to be centered on smallest diagonal value to be correct
    #assert 2*N+1 > mindim(nu, eta), "Dimension too small for accurate eigenvalues."

    k = np.arange(2*N+1)-N
    A = np.diag((2*k+nu)**2).astype(float)
    A += np.diag([eta]*(2*N), k=1)
    A += np.diag([eta]*(2*N), k=-1)

    r = np.linalg.eig(A)
    a = r.eigenvalues
    c = r.eigenvectors # eigenvectors in columns, i.e. along first index

    # sort in ascending eigenvalues
    s = np.argsort(a)
    c, a = c[:,s], a[s]

    # construct functions
    # 1. axis: z, 2. axis: eigenvalues, 3. axis: k (summed)
    psi = np.exp(1j*nu*z[:,None]) * np.sum(c.T[None,:,:] * np.exp(1j*2*k[None,None,:]*z[:,None,None]), axis=-1)

    return a, c, psi


def cpb_hamiltonian_eigenstates(ng, EJoverEC, Npoints, Nstates):
    r"""Calculate eigenstates and eigenenergies of the CPB.

    $$H = 4 E_C (\hat n - n_g)^2 - E_J \cos \phi $$

    Eigenstates are expressed in phase basis psi(phi) = <phi|psi>.

    Note: $\hat n$ is the Cooper pair number, thus the factor 4
    in front of $E_C$.

    Parameters
    ----------
    ng : float
        Offset charge number.
    EJoverEC : float
        Josephson energy EJ in units of charging energy EC.
    Npoints : int
        Number of phi points in interval [-pi to pi).
    Nstates : int
        Number of eigenstates to return.

    Returns:
    -------
    E : numpy.ndarray of shape (Nstates,)
        Energies E (sorted ascending) in units of EC.
    psi : numpy.ndarray of shape (Npoints,Nstates)
        Eigenstates (along second axis) corresponding to energies.
        First axis is evaluation at phi.
    phi : numpy.ndarray of shape (Npoints,)
        phi values in [-pi, +pi).
    """
    phi = np.linspace(-np.pi, np.pi, Npoints+1)[:-1]

    # ng + nu/2 is integer for 2pi periodic solutions.
    # Use 2 periodicity in nu to have nu close to zero
    # for better precision.
    nu = -2*ng + 2*int(ng)

    eta = -EJoverEC/2
    # Calculates 2*Nstates+1 solutions
    E, _, psi = Mathieu_solutions(phi/2, nu, eta, Nstates)

    # normalize wavefunctions
    psi = psi / np.sqrt(2*np.pi)

    return E[:Nstates], psi[:,:Nstates], phi


def cpb_cavity_hamiltonian(EJoverEC, ng, ERoverEC, coverEC, J, N, npoints=500):
    r"""Eigenenergies and -states of CPB coupled to cavity.

    Implements eq. (3.2) from Koch et al.,  Phys Rev. A (2007):

    $$H = E_j | j \rangle \langle j | + E_r a^\dagger a + \sum_{ij} g_{ij} | i \rangle \langle j | (a + a^\dagger)$$

    $| j \rangle$ are the eigenstates of the uncoupled CPB with energies $E_j$,
    calculated using :code:`cpb_hamiltonian_eigenstates()`.

    $a^\dagger$ are ladder operators of the cavity photon number.

    $g_{ij} = c \langle i | \hat n | j \rangle$ is the coupling, where
    $\hat n$ is the charge operator in phase basis, i.e.
    $\hat n | \psi \rangle = -i \frac{\rm d}{\rm d \phi} \langle \phi | \psi \rangle$.
    The coupling prefactor :math:`c=2\beta e V_\text{rms}^0` determines
    coupling strength independent of CPB states.

    Parameters
    ----------
    EJoverEC : float
        CPB Josephson energy in units of charging energy
    ng : float
        Offset charge (In number of Cooper pairs)
    ERoverEC : float
        Cavity resonance, i.e. photon energy $\hbar \omega_r$, in units of charging energy.
    coverEC : float
        Coupling prefactor c, also in units of charging energy.
    J : int
        Number of qubit states
    N : int
        Number of cavity states
    npoints : int
        Number of points on $\phi = [-\pi, pi)$ to evaluate wavefunction on.
        Needs to be large enough to correctly form derivative of wavefunction to
        evaluate charge operator $\hat n$.

    Returns
    -------
    e : numpy.ndarray of shape (J*N,)
        Eigenvalues sorted in ascending order, i.e. eigenenergies in
        units of EC.
    v : numpy.ndarray of shape (J, N, J*N)
        Eigenvectors (corresponding to eigenvalues along last axis)
        with weights in bare qubit states (along first axis) and weights
        in bare cavity states (along second axis).
    """
    # cavity ladder operators
    a = np.diag(np.sqrt(np.arange(1, N)), k=1)
    adag = np.conj(a).T

    # qubit
    Eq, psi, phi = cpb_hamiltonian_eigenstates(ng, EJoverEC, npoints, J)

    # n |j> in phase space: -i d/dphi <phi|j>
    # Note: sample spacing in phi cancels out when doing <i|n|j> next
    npsi = -1j * np.gradient(psi, axis=0)
    # <i | n | j>
    charge_coupling = np.array([[np.sum(np.conj(psi[:,i])*npsi[:,j]) for j in range(len(Eq))] for i in range(len(Eq))])
    # g = g_{ij} |i><j|
    # Discard non-hermitian part arising due to numerical errors
    g = coverEC * (charge_coupling + np.conj(charge_coupling).T) / 2

    # putting it together
    Hq = np.diag(Eq) # J x J
    Hr = ERoverEC * (adag @ a) # N x N
    Hcoupling =  np.tensordot(g, a + adag, axes=0) # J x J x N x N
    H = np.tensordot(Hq, np.identity(N), axes=0) + np.tensordot(np.identity(J), Hr, axes=0) + Hcoupling
    # Now have tensor of J x J x N x N
    # Want matrix of (J*N) x (J*N)
    H = np.moveaxis(H, [0,1,2,3], [0,2,1,3]).reshape(J*N,J*N)

    assert np.allclose(0, np.conj(H).T - H), "H should be hermitian"
    r = np.linalg.eig(H)
    e = r.eigenvalues.real
    v = r.eigenvectors.real

    s = np.argsort(e)
    e, v = e[s], v[:,s]

    return e, v.reshape(J,N,J*N)


def cpb_cavity_identify_energies(e, v):
    """
    Identify energies of coupled system with state indices
    of uncoupled system.

    For example the qubit transition is :code:`y[1,0]-y[0,0]` with
    no photons in the cavity (y is the output of this function).
    The coupled cavity frequency with the qubit in ground state is
    :code:`y[0,1]-y[0,0]`.

    e and v should be output of :code:`cpb_cavity_hamiltonian()`.

    Works only when states are close to uncoupled states, i.e.
    small coupling or dispersive limit. For strongly mixed states it
    is not possible to assign corresponding uncoupled states.
    This might result in wrong assignments and the output containing
    NaNs.

    Parameters
    ----------
    e : numpy.ndarray of shape (J*N)
        Coupled system energies
    v : numpy.ndarray of shape (J, N, J*N)
        Weights of states in uncoupled system

    Returns
    -------
    energies : numpy.ndarray of shape (J, N)
        Element i,n gives energy of coupled system with
        qubit state i and cavity photon number n.
    """
    # index (i,n) with largest weight in coupled system
    maxidx = np.unravel_index(np.argmax(v.reshape(v.shape[0]*v.shape[1], v.shape[2]), axis=0), v.shape[:2])
    # assign into (J,N) array
    energies = np.full(v.shape[:2], np.nan)
    energies[maxidx[0][::-1], maxidx[1][::-1]] = e[np.arange(v.shape[2])[::-1]]
    return energies
