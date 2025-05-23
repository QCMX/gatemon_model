# -*- coding: utf-8 -*-
"""
Numerically solve Hamiltonians for qubits with arbitrary potentials and ABS potential.

Generalization of the code for the Cooper pair box.
"""

import numpy as np


def qubit_arbitraryV_eigenstates(ng, Vfunc, Nstates, phi=None):
    r"""
    Finds energies and wavefunctions for superconducting qubit Hamiltonian with arbitrary periodic potential.

    The Hamiltonian is expressed in units of charging energy $E_C$:

    $$H / E_C = 4 (\hat n - n_g)^2 + V(\hat \varphi) / E_C.$$

    The potential $V(\varphi)$ is assumed to be $2\pi$-periodic in [-pi, pi).

    Wavefunctions are normalized in the $2\pi$ interval:

    $$\int_{-\pi}^\pi \psi^*(\varphi)\, \psi(\varphi)\, \mathrm d \varphi = \sum_n c_n^* c_n = 1,$$

    where $c_n$ are the coefficients of the wavefunction in charge space
    (called :code:`jn` below).

    $$\psi(\varphi) = \frac{1}{\sqrt{2\pi}} \sum_n c_n e^{i n \varphi}.$$

    Beware of truncation errors in the highest energy states, or for
    potentials with high freq. Fourier components.

    Parameters
    ----------
    ng : float
        Offset charge (number of Cooper pairs).
    Vfunc : float -> float
        Potential term V(phi) in units of charging energy.
        Will be evaluated on 2*Nstates+1 points in [-pi, pi).
    Nstates : int
        Matrix size to use for solution.
    phi : numpy.ndarray of shape (Npoints,), optional
        Values of phi to evaluate wavefunctions on.
        If None, wavefunctions are not evaluated in phase space.
        Default is None.

    Returns
    -------
    e : numpy.ndarray of shape (Nstates,)
        Eigenenergies in units of charging energy
    jn : numpy.ndarray of shape (Nstates,Nstates)
        Eigenfunctions in charge space
    n : numpy.ndarray of shape (Nstates)
        Charge number n associated to each charge state
    psi : numpy.ndarray of shape (Npoints,Nstates), optional.
        Eigenfunctions psi(phi) in phase space.
        Not present if phi is None.
    """
    N = 2*Nstates+1
    phiV = np.linspace(-np.pi, np.pi, N+1)[:-1]
    V = Vfunc(phiV)

    m = np.arange(N) - N//2
    vm = np.sum(V[None,:] * np.exp(-1j * phiV[None,:] * m[:,None]), axis=1) / N

    ## Optional: V symmetric, so vm real
    #assert np.allclose(vm.imag, 0)
    #vm = vm.real

    # V as matrix operator: sum_m v_m |n+m><n|
    n = np.arange(Nstates) - Nstates//2
    Vop = np.zeros((Nstates, Nstates)).astype(complex)
    for i in range(Nstates):
        Vop[i] = vm[i+1:i+1+Nstates][::-1]

    # put together Hamiltonian
    q = -2*ng
    H = np.diag((2*n+q)**2) + Vop
    assert np.allclose(0, np.conj(H).T - H), "H should be Hermitian"

    # solve eigenvalue problem
    r = np.linalg.eig(H)
    e = r.eigenvalues
    c = r.eigenvectors

    assert np.allclose(e.imag, 0), "Eigenvalues should be real since H is Hermitian"
    e = e.real

    # sort in ascending eigenvalues
    s = np.argsort(e)
    c, e = c[:,s], e[s]

    if phi is not None:
        psi = np.sum(c.T[None,:,:] * np.exp(1j * n[None,None,:] * phi[:,None,None]), axis=-1) / (2*np.pi)**0.5
        return e, c, n, psi
    return e, c, n


def charge_transitions(jn, c=1):
    r"""Matrix elements for transitions excited via qubit charge (ie capacitively).

    $g_{ij} = c \langle i | \hat n | j \rangle$

    The prefactor :math:`c=2\beta e V_\text{rms}^0` determines
    coupling strength.

    Parameters
    ----------
    jn : numpy.nparray of shape (M, J)
        Qubit wavefunction in charge basis.
        Assumes charge indices :code:`n=range(M)-M//2`.
        Thus compatible with output of :code:`qubit_arbitraryV_eigenstates()`.
    c : float
        Coupling prefactor.

    Returns
    -------
    g : numpy.ndarray of shape (J, J)
        Coupling matrix elements. Is a Hermitian matrix.
        In same units as c.
    """
    J = jn.shape[1]
    n = np.arange(jn.shape[0]) - jn.shape[0]//2
    # g_{ij} = <i| n |j>
    # g = g_{ij} |i><j|
    charge_coupling = np.array([[np.sum(np.conj(jn[:,i])*n*jn[:,j]) for j in range(J)] for i in range(J)])
    # Discard non-hermitian part arising due to numerical errors
    g = c * (charge_coupling + np.conj(charge_coupling).T) / 2
    return g


def estimate_minNstates(ng, Vfunc, rtol=1e-5, startNstates=4, stepNstates=2, maxsteps=20, testfunc=None):
    r"""
    Estimate minimum matrix size needed for accurate calculation of qubit transition (E01).

    Iteratively increases matrix size until relative error between consecutive
    E01 is smaller than rtol.

    Note that most likely higher transitions will have larger numerical error.

    Parameters
    ----------
    ng : float
        Offset charge.
    Vfunc : Vfunc : float -> float
        Potential term V(phi) in units of charging energy.
        Will be evaluated on 2*Nstates+1 points in [-pi, pi).
    rtol : float
        Maximum relative error.
    startNstates : int
        Initial matrix size.
    stepNstates : int
        How much to increase matrix size for each step.
    maxsteps : int
        Maximum number of tries. If the desired tolerance is not
        reached, returns None.
    testfunc : np.ndarray -> float
        Function to test for relative change.
        If None, uses qubit frequency (difference between lowest two energies).
        Default None.

    Returns
    -------
    Nstates : int or None
        Minimum matrix size for desired tolerance on qubit transition.
        None if the tolerance could not be reached within `maxsteps`.
    """
    if testfunc is None:
        testfunc = lambda e: e[1] - e[0]
    e = qubit_arbitraryV_eigenstates(ng, Vfunc, startNstates)[0]
    v1 = testfunc(e)
    for i in range(1, maxsteps):
        e2 = qubit_arbitraryV_eigenstates(ng, Vfunc, startNstates + i*stepNstates)[0]
        v2 = testfunc(e2)
        if np.abs(v2-v1) <= rtol*np.abs(v2):
            return startNstates + (i-1)*stepNstates
        v1 = v2
    return None


def lowerABS(phi, NDelta, tau):
    r"""
    Energy of lower Andreev bound state in atomic contact model with homogeneous transmission.

    Note that $E_J=\Delta N \tau / 4$. With effective
    number of channels $N$ and same transmission $\tau$
    for all channels. $2\Delta$ is the superconducting gap.

    Parameters
    ----------
    phi : float or array
        Superconducting phase difference
    NDelta : float or array
        Gap times effective channel number
    tau : float or array
        Transmission in [0, 1].

    Returns
    -------
    EA : float or array
        Energy of lower Andreev bound state in same units as EJ.
    """
    return -NDelta * np.sqrt(1 - tau * np.sin(phi/2)**2)


def gatemon_lowerABS_eigenstates(ng, NDelta, tau, Nstates=None, phi=None):
    r"""
    Eigenenergies and states of gatemon with homegeneous transmission.

    Plugs :code:`lowerABS()` as potential term for :code:`qubit_arbitraryV_eigenstates()`.

    Parameters
    ----------
    ng : float
        Offset charge.
    NDelta : float
        Gap times effective channel number.
    tau : float
        Transmission, same for all channels.
    Nstates : int or None
        Size of Hamiltonian to solve numerically.
        If None, uses :code:`estimate_minNstates()`.
    phi : numpy.ndarray or None
        Points to evaluate wavefunction at.
        Default is None.

    Returns
    -------
    Same as :code:`qubit_arbitraryV_eigenstates()`.
    """
    Vfunc = lambda phi: lowerABS(phi, NDelta, tau)
    if Nstates is None:
        Nstates = estimate_minNstates(ng, Vfunc)
        if Nstates is None:
            raise Exception("Could not determine Nstates automatically.")
    return qubit_arbitraryV_eigenstates(ng, Vfunc, Nstates, phi)


def qubit_cavity_hamiltonian(Ej, jn, Er, c, N):
    r"""Eigenenergies and -states of qubit coupled to cavity.

    Implements eq. (3.2) from Koch et al.,  Phys Rev. A (2007):

    $$H = E_j | j \rangle \langle j | + E_r a^\dagger a + \sum_{ij} g_{ij} | i \rangle \langle j | (a + a^\dagger)$$

    $| j \rangle$ are the eigenstates of the uncoupled CPB with energies $E_j$,
    calculated using. Here represented via $| j \rangle = \sum_n j_n | n \rangle$.

    $a^\dagger$ are ladder operators of the cavity photon number.

    $g_{ij} = c \langle i | \hat n | j \rangle$ is the coupling.
    The prefactor :math:`c=2\beta e V_\text{rms}^0` determines
    coupling strength.

    Parameters
    ----------
    Ej : numpy.nparray of shape (J,)
        Qubit eigenenergies.
    jn : numpy.nparray of shape (M, J)
        Qubit wavefunction in charge basis.
        Assumes charge indices :code:`n=range(M)-M//2`.
        Thus compatible with output of :code:`qubit_arbitraryV_eigenstates()`.
    Er : float
        Cavity resonance, i.e. photon energy $\hbar \omega_r$.
    c : float
        Coupling prefactor.
    N : int
        Number of cavity states.

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
    # qubit transition matrix element
    J = Ej.size
    n = np.arange(jn.shape[0]) - jn.shape[0]//2
    # <i| n |j>
    charge_coupling = np.array([[np.sum(np.conj(jn[:,i])*n*jn[:,j]) for j in range(J)] for i in range(J)])
    # g = g_{ij} |i><j|
    # Discard non-hermitian part arising due to numerical errors
    g = c * (charge_coupling + np.conj(charge_coupling).T) / 2

    # cavity ladder operators
    a = np.diag(np.sqrt(np.arange(1, N)), k=1)
    adag = np.conj(a).T

    # putting it together
    Hq = np.diag(Ej) # J x J
    Hr = Er * (adag @ a) # N x N
    Hcoupling =  np.tensordot(g, a + adag, axes=0) # J x J x N x N
    H = np.tensordot(Hq, np.identity(N), axes=0) + np.tensordot(np.identity(J), Hr, axes=0) + Hcoupling
    # Now have tensor of J x J x N x N
    # Want matrix of (J*N) x (J*N)
    H = np.moveaxis(H, [0,1,2,3], [0,2,1,3]).reshape(J*N,J*N)

    assert np.allclose(0, np.conj(H).T - H), "H should be Hermitian"
    r = np.linalg.eig(H)
    e = r.eigenvalues.real
    v = r.eigenvectors.real

    assert np.allclose(e.imag, 0), "Eigenvalues should be real since H is Hermitian"
    e = e.real

    s = np.argsort(e)
    e, v = e[s], v[:,s]

    return e, v.reshape(J,N,J*N)


def qubit_cavity_identify_energies(e, v):
    """
    Identify energies of coupled system with state indices of uncoupled system.

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

    Same function as :code:`gatemon.cooper_pair_box.qubit_cavity_identify_energies()`.

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
