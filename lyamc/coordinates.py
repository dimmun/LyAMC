import numpy as np

# Pauli matrices
id_m = np.identity(2).reshape(1, 2, 2)
S_x = np.array(((0, 1), (1, 0))).reshape(1, 2, 2)
S_y = np.array(((0, -1j), (1j, 0))).reshape(1, 2, 2)
S_z = np.array(((1, 0), (0, -1))).reshape(1, 2, 2)


# In what follows, everything is written to track n scatterings at once
# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def vs_to_quat(vs):
    """Converts array of velocities to boost quaternions (boost into frame
    with v)
    Parameters:
        vs: nx3 array of velocities (in units of c), where n is ray index
    Outputs:
        nx2x2 matrix representing the boost quaternions
    """
    # compute prefactor involving lorentz gamma
    prefac = (1.0 / (1.0 + np.sqrt(1.0 - np.sum(vs * vs, axis=-1)))). \
        reshape(-1, 1, 1)
    vq = -1j * (vs[:, 0].reshape(-1, 1, 1) * S_z +
                vs[:, 1].reshape(-1, 1, 1) * S_y +
                vs[:, 2].reshape(-1, 1, 1) * S_x)
    outq = prefac * vq
    return outq


def ns_to_quat(ns):
    """Converts array of input direction cosines to null-ray quaternions
    Parameters:
        ns: nx3 array of direction cosines, where n is ray index
    Outputs:
        nx2x2 matrix representing the null-ray quaternions
    """
    outq = 1j * (ns[:, 0].reshape(-1, 1, 1) * S_z +
                 ns[:, 1].reshape(-1, 1, 1) * S_y +
                 ns[:, 2].reshape(-1, 1, 1) * S_x)
    return outq


def quat_to_ns(quats):
    """Converts array of null-ray quaternions to direction cosines
    Parameters:
        quats: nx2x2 matrix representing the null-ray quaternions, where n is
               ray index
    Outputs:
        nx3 array of direction cosines
    """
    # Project out components
    nx = np.real(
        -0.5 * 1j * np.trace(np.matmul(S_z, quats), axis1=-2, axis2=-1))
    ny = np.real(
        -0.5 * 1j * np.trace(np.matmul(S_y, quats), axis1=-2, axis2=-1))
    nz = np.real(
        -0.5 * 1j * np.trace(np.matmul(S_x, quats), axis1=-2, axis2=-1))
    ns = np.c_[nx, ny, nz]
    # Normalize to make sure
    norms = np.linalg.norm(ns, axis=-1).reshape(-1, 1)
    ns = ns / norms
    return ns


def ns_to_rot(ns):
    """Converts array of input direction cosines to rotation quaternions
    Parameters:
        ns: nx3 array of direction cosines, where n is ray index
    Outputs:
        nx2x2 matrix representing the rotation quaternions
    """
    # Normalize to be safe
    norms = np.linalg.norm(ns, axis=-1)
    mus = (ns[:, 2] / norms).reshape(-1, 1, 1)
    phib2 = 0.5 * np.arctan(ns[:, 1] / ns[:, 0])
    # Fix correct branch
    phib2[ns[:, 0] < 0] += 0.5 * np.pi
    phib2 = phib2.reshape(-1, 1, 1)
    # First Euler rotation
    phiq = np.cos(phib2) * id_m - 1j * np.sin(phib2) * S_x
    # Second Euler rotation
    thetaq = (np.sqrt(0.5 * (1.0 + mus)) * id_m -
              1j * np.sqrt(0.5 * (1.0 - mus)) * S_y)
    outq = np.matmul(thetaq, phiq)
    return outq


# ---------------------------------------------------------------------
# Main functions
# ---------------------------------------------------------------------
def samplephase(n):
    """Function to generate n rays sampling the scattering phase function,
    assumed to be ~ (1 + \mu^2)
    Parameters:
        n: number of rays
    Outputs:
        nx3 array of direction cosines (relative to incoming rays on z)
    """
    rvs = np.random.random((n, 2))
    phis = rvs[:, 1] * 2.0 * np.pi
    sqfact = np.power(- 2.0 + 4.0 * rvs[:, 0] +
                      np.sqrt(5.0 - 16.0 * rvs[:, 0] + 16.0 * rvs[:, 0] ** 2),
                      1.0 / 3.0)
    # Invert CDF
    mus = sqfact - 1 / sqfact
    sins = np.sqrt(1.0 - mus ** 2)
    ns = np.c_[sins * np.cos(phis), sins * np.sin(phis), mus]
    return ns


def scattering_lab_frame(freqs, ns, vs):
    """Function to perform one scattering for n rays off n atoms
    Parameters:
        freqs: array with n frequencies of input rays in units of hydrogen mass
        ns:    nx3 array with lab-frame direction cosines of input rays
        vs:    nx3 array with velocities of atoms (in units of c)
    Outputs:
        array with n frequencies of output rays in units of hydrogen mass
        nx3 array with lab-frame direction cosines of output rays
    """
    # Define initial null-ray quaternion
    q_n = ns_to_quat(ns)

    # Define boost quaternion into atom rest frame
    q_b = vs_to_quat(vs)
    # Lorentz factor for boost
    g_b = 1.0 / np.sqrt(1.0 - np.sum(vs * vs, axis=-1))

    # Photon energy in atom rest frame
    freqs_arf = g_b * freqs * (1.0 - np.sum(ns * vs, axis=-1))
    # Null-ray quaternion in atom rest frame
    q_n_arf = np.matmul(q_n + q_b, np.linalg.inv(1.0 - np.matmul(q_b, q_n)))
    # Direction cosine in atom rest frame
    ns_arf = quat_to_ns(q_n_arf)

    # Define rotation quaternion to put photon on the z-axis
    q_rot_euler = ns_to_rot(ns_arf)

    # Go into center of mass frame
    # First define velocities for boost
    vzs_com = freqs_arf / (1.0 + freqs_arf)
    vs_to_com = np.c_[np.zeros((len(freqs), 2)), vzs_com]
    # Lorentz factor for boost
    g_b_com = 1.0 / np.sqrt(1.0 - np.sum(vs_to_com * vs_to_com, axis=-1))
    # Photon energy in com frame
    freqs_com = g_b_com * freqs_arf * (1.0 - vzs_com)
    # Finally, boost quaternion into center of mass frame
    q_b_com = vs_to_quat(vs_to_com)

    # Now sample outgoing rays in center of mass frame
    ns_out_com = samplephase(len(freqs))
    # Define corresponding null-ray quaternions
    qs_out_com = ns_to_quat(ns_out_com)

    # Now boost back into incoming atom rest frame
    q_out_arf = np.matmul(qs_out_com - q_b_com,
                          np.linalg.inv(1.0 + np.matmul(q_b_com, qs_out_com)))
    # Outgoing photon energy in incoming atom rest frame
    freqs_out_com = (g_b_com * freqs_com *
                     (1.0 + np.sum(vs_to_com * ns_out_com, axis=-1)))

    # Now undo rotation done to put photon on the z-axis
    q_out_uneuler = np.matmul(np.matmul(np.linalg.inv(q_rot_euler), q_out_arf),
                              q_rot_euler)
    # new direction cosines of photon
    ns_out_uneuler = quat_to_ns(q_out_uneuler)

    # Finally, boost back into lab frame
    q_out_lab = np.matmul(q_out_uneuler - q_b,
                          np.linalg.inv(1.0 + np.matmul(q_b, q_out_uneuler)))
    # compute energies
    freqs_out_lab = (g_b * freqs_out_com *
                     (1.0 + np.sum(vs * ns_out_uneuler, axis=-1)))
    # compute directions
    ns_out_lab = quat_to_ns(q_out_lab)

    return freqs_out_lab, ns_out_lab


pass
