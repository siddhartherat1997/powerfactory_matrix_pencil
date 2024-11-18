# pencil.py

import numpy as np
import pandas as pd
from math import ceil
from scipy.linalg import hankel, svd

def Pencil(Q, Ts):
    """
    Perform the Pencil method on the input data to estimate system frequencies, damping ratios, and magnitudes.

    Parameters:
    Q (numpy array): Input signal data (time series).
    Ts (float): Sampling period (time step between data points).

    Returns:
    pandas DataFrame: A sorted dataframe with estimated frequency, damping ratio, eigenvalues, and magnitude.
    """
    # Length of the input signal
    N = len(Q)
    
    # Determine the pencil parameter and create Hankel matrix
    l = ceil(len(Q) / 2)  # pencil parameter = range N/2 to 2N/3
    Y = hankel(Q[0:N-l], Q[N-l-1:N])
    
    # Singular value decomposition
    u, s, vh = svd(Y, compute_uv=True, full_matrices=True)
    smat = np.zeros((u.shape[1], vh.shape[0]), dtype=complex)
    smat[:len(s), :len(s)] = np.diag(s)

    # Rank determination using singular values
    rank = sum(s / max(s) > 1e-7)
    
    # Create new matrices with truncated singular values and vectors
    V = np.conjugate(np.transpose(vh))
    s_new = smat[:, :rank]
    v_new = V[:, :rank]

    # Creating vectors v1 and v2
    v1, v2 = v_new[:-1].T, v_new[1:].T
    Y1, Y2 = u @ s_new @ v1, u @ s_new @ v2
    
    # Perform the generalized least squares estimation
    dfil = np.linalg.pinv(Y1) @ Y2
    z1, _ = np.linalg.eig(dfil)
    z = z1[:rank]

    # Calculate damping ratios and frequencies from eigenvalues
    zabs, zangle = np.abs(z), np.angle(z)
    zeta = (np.log(zabs) / Ts)[:rank]
    freq = (zangle / (2 * np.pi * Ts))[:rank]
    zp = zeta / np.sqrt(zeta**2 + (2 * np.pi * freq)**2)

    # Construct the matrix Z with powers of the eigenvalues
    Z = np.zeros((N, rank), dtype=complex)
    for i in range(N):
        Z[i, :] = pow(z, i)

    epsilon = 1e-10
    B = np.linalg.inv(Z.T @ Z + epsilon * np.eye(Z.shape[1])) @ Z.T @ Q
    mag = 2.0 * np.abs(B)

    # Create a DataFrame with the results
    df = pd.DataFrame({
        "Frequency (Hz)": freq,
        "zeta (%)": zp,
        "Eigen (discrete)": z[:rank],
        "Magnitude": mag
    })

    # Filter the results to include frequencies between 2.5 and 3.8 Hz
    df_filtered = df[(2.5 <= abs(df["Frequency (Hz)"])) & (abs(df["Frequency (Hz)"]) <= 3.8)]
    
    # Return the sorted DataFrame by Magnitude
    return df_filtered.sort_values(by=['Magnitude'], ascending=False).reset_index(drop=True)

