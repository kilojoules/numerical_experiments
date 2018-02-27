import numpy as np
import matplotlib.pyplot as plt
from solver import geturec, xf

nx = 7000
x = np.linspace(0, xf, nx)
u_record = geturec(nu=0.005, evolution_time = 1, x=x)

# SVD the covariance matrix
psi, D, phi = np.linalg.svd(u_record) # P D Q is the approximation

# choose # of modes to keep
MODES = 2

S = np.zeros((nx, u_record.shape[1]))
mm = min(nx, u_record.shape[1])
S[:mm, :mm] = np.diag(D)
plt.plot(np.sqrt(D)) ; plt.yscale('log') ; plt.savefig('spectrum.pdf') ; plt.clf()
assert(np.allclose(u_record, np.dot(psi, np.dot(S, phi)))) # check that a = P D Q
phi = phi.T
Q = np.dot(S[:, :MODES], phi[:, :MODES].T)

# Normalize the basis functions
for ss in range(phi.shape[1]):
    phi[:, ss] /= np.dot(phi[:, ss], phi[:, ss])
for ss in range(psi.shape[1]):
    psi[:, ss] /= np.dot(psi[:, ss], psi[:, ss])

# plot the modes
colors = plt.cm.coolwarm(np.arange(MODES) / (MODES-1.))
for i in range(MODES): plt.plot(Q[i, :], c=colors[i])
plt.savefig('Q_Modes.pdf') ; plt.clf()

# reconstruct it!
u_recon = np.dot(psi, Q)

# Plot the reconstruction
plt.plot(x, u_record[:, -1], c='k')
plt.plot(x, u_record[:, 0], c='gray')
plt.plot(x, u_recon[:, 0], c='darkred', ls='--')
plt.plot(x, u_recon[:, -1], c='r', ls='--')
plt.xlabel('x', size=20)
plt.ylabel('u', size=20)
plt.savefig('PODReconstruct.pdf')
plt.clf()

# create new full-order solution using POD IC
u_pert = geturec(u0=u_recon[:, 0], nu=0.005, evolution_time = 1, x=x)

# compare the POD-perutbred with the baseline
plt.plot(x, u_pert[:, 0], label="POD Perturbation Initial")
plt.plot(x, u_pert[:, -1], label="POD Perturbation Final")
plt.plot(x, u_record[:, 0], c='gray', label="Baseline Initial")
plt.plot(x, u_record[:, -1], c='k', label="Baseline Final")
lgd = plt.legend(loc='lower center', bbox_to_anchor=[.5, 1])
plt.savefig('PODperturbed.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.clf()

# Compare the 1st POD mode with the average flow field
f, ax = plt.subplots()
ax2 = ax.twinx()
ax.plot(psi[:, 0], c='r')
ax2.plot(-1 * u_record.mean(1), ls='--')
plt.savefig('./firstModeVersusMean.pdf')
plt.clf()
