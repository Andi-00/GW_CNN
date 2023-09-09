## Code to generate the 10^4 parameters, strains and spectrograms

import sys
import os

from gwpy.timeseries import TimeSeries

import matplotlib.pyplot as plt
import numpy as np

from few.trajectory.inspiral import EMRIInspiral
from few.amplitude.romannet import RomanAmplitude
from few.amplitude.interp2dcubicspline import Interp2DAmplitude
from few.waveform import FastSchwarzschildEccentricFlux, SlowSchwarzschildEccentricFlux, GenerateEMRIWaveform
from few.utils.utility import (get_overlap,
                               get_mismatch,
                               get_fundamental_frequencies,
                               get_separatrix,
                               get_mu_at_t,
                               get_p_at_t,
                               get_kerr_geo_constants_of_motion,
                               xI_to_Y,
                               Y_to_xI)

from few.utils.ylm import GetYlms
from few.utils.modeselector import ModeSelector
from few.summation.interpolatedmodesum import CubicSplineInterpolant
from few.waveform import SchwarzschildEccentricWaveformBase
from few.summation.interpolatedmodesum import InterpolatedModeSum
from few.summation.directmodesum import DirectModeSum
from few.utils.constants import *
from few.summation.aakwave import AAKSummation
from few.waveform import Pn5AAKWaveform, AAKWaveformBase



use_gpu = False

# keyword arguments for inspiral generator (RunSchwarzEccFluxInspiral)
inspiral_kwargs={
        "DENSE_STEPPING": 0,  # we want a sparsely sampled trajectory
        "max_init_len": int(1e3),  # all of the trajectories will be well under len = 1000
    }

# keyword arguments for inspiral generator (RomanAmplitude)
amplitude_kwargs = {
    "max_init_len": int(1e3),  # all of the trajectories will be well under len = 1000
    "use_gpu": use_gpu  # GPU is available in this class
}

# keyword arguments for Ylm generator (GetYlms)
Ylm_kwargs = {
    "assume_positive_m": False  # if we assume positive m, it will generate negative m for all m>0
}

# keyword arguments for summation generator (InterpolatedModeSum)
sum_kwargs = {
    "use_gpu": use_gpu,  # GPU is availabel for this type of summation
    "pad_output": False,
}

# THE FOLLOWING THREAD COMMANDS DO NOT WORK ON THE M1 CHIP, BUT CAN BE USED WITH OLDER MODELS
# EVENTUALLY WE WILL PROBABLY REMOVE OMP WHICH NOW PARALLELIZES WITHIN ONE WAVEFORM AND LEAVE IT TO
# THE USER TO PARALLELIZE FOR MANY WAVEFORMS ON THEIR OWN.

# set omp threads one of two ways
# num_threads = 4

# this is the general way to set it for all computations
# from few.utils.utility import omp_set_num_threads
# omp_set_num_threads(num_threads)

few = FastSchwarzschildEccentricFlux(
    inspiral_kwargs=inspiral_kwargs,
    amplitude_kwargs=amplitude_kwargs,
    Ylm_kwargs=Ylm_kwargs,
    sum_kwargs=sum_kwargs,
    use_gpu=use_gpu,
    # num_threads=num_threads,  # 2nd way for specific classes
)

plt.rcParams['pgf.rcfonts'] = False
plt.rcParams['font.serif'] = []
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = True
plt.rcParams['axes.formatter.useoffset'] = False
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['errorbar.capsize'] = 2
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['legend.title_fontsize'] = 16
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.1

#plt.rcParams['savefig.transparent'] = True
plt.rcParams['figure.figsize'] = (8, 5)


gen_wave = GenerateEMRIWaveform("Pn5AAKWaveform")

# parameters
T = 0.05 # years
dt = 5 # seconds

M = 1e4  # solar mass
mu = 1  # solar mass

dist = 1.0  # distance in Gpc

p0 = 12.0
e0 = 0.3
x0 = 0.99  # will be ignored in Schwarzschild waveform

qS = 1E-6  # polar sky angle
phiS = 0.0  # azimuthal viewing angle


# spin related variables
a = 0.6  # will be ignored in Schwarzschild waveform
qK = 1E-6  # polar spin angle
phiK = 0.0  # azimuthal viewing angle


# Phases in r, theta and phi
Phi_phi0 = 0
Phi_theta0 = 0
Phi_r0 = 0

# Generate the random parameters for the EMRIs
# The parameters include M, mu / d, a, e0 and p0

def gen_parameters(N):
    n = np.random.uniform(4, 7, N)
    M = 10 ** n
    dm = np.random.uniform(1, 1E2, N)
    a = np.random.uniform(0, 1, N)
    e0 = np.random.uniform(0.1, 0.7, N)
    p0 = np.random.uniform(10, 16, N)

    parameters = np.array([np.array([M[i], dm[i], a[i], e0[i], p0[i]]) for i in range(N)])

    return parameters

# Generate the strain h from the parameters par and returns it
def gen_strain(par):
    M = par[:, 0]
    mu = np.ones_like(M)
    d = par[:, 1]
    a = par[:, 2]
    e0 = par[:, 3]
    p0 = par[:, 4]

    h = [gen_wave(M[i], mu[i], a[i], p0[i], e0[i], x0, d[i], qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0, T=T, dt=dt) for i in range(len(d))]

    return h

# Generate the spectrograms of the given strains
def gen_specs(hs):

    specs = []

    for h in hs:
        
        # Fill the strain array with zeros, so that all have the same length
        # independant of the signal duration
        hp = np.pad(h.real, (0, 315582 - len(h)))

        ts = TimeSeries(hp, dt = dt)

        data = ts.spectrogram(2E4) ** (1/2)


        specs.append(data)

    return np.array(specs)

# Save the spectrogram files as csv data
def save_files(files, loc, n = 0):
    for i in range(len(files)):
        np.savetxt("./thesis_plots/plots/spec_{}.png".format(i), np.array(files[i]), delimiter = ",")




# Save the parameters in a csv file
p0 = np.array([1E5, 1, 0.6, 0.3, 12])
p1 = np.copy(p0)
p2 = np.copy(p0)
p3 = np.copy(p0)


rM = [1E4, 1E5, 1E6]
rd = [1, 5, 10]
ra = [0.2, 0.4, 0.6, 0.8]
re = [0.1, 0.3, 0.5, 0.7]
rp = [10, 12, 14, 16]

p1[2] = 0.2
p2[3] = 0.7
p3[4] = 16

par = [p0, p1, p2, p3]

# for r in rM:
#     p0[0] = r
#     par.append(p0)

par = np.array(par)

h = gen_strain(par)

specs = gen_specs(h)

# for i in len(specs):
#     np.savetxt("./thesis_plots/plots/chapter_6/M/spec_{:.1E}.csv".format(rM[i]), np.array(specs[i]), delimiter = ",")


# fig, ax = plt.subplots()

# h = h[0][: 400].real * 1E23
# t = np.arange(len(h)) * dt

# ax.plot(t, h, color = "#e60049")
# ax.set_xlabel("Time $t$ / s")
# ax.set_ylabel("Strain $h_+ / 10^{-23}$")
# ax.grid(True)

# plt.savefig("./thesis_plots/plots/chapter_6/reference_h.png")


import matplotlib.colors as colors

names = ["reference", "a", "e", "p"]

for i in range(len(par)):

    fig, ax = plt.subplots()

    x = (np.arange(0, 79) + 0.5) * 2E4 / (3600 * 24)
    y = (np.arange(2E3 + 1) + 0.5) * 5E-5
    z = np.swapaxes(specs[i], 0, 1)


    pc = ax.pcolormesh(x, y, z, norm = colors.LogNorm(vmin = np.max(z) * 1E-6, vmax = np.max(z)))
    ax.set_yscale("log")

    ax.set_ylim(1E-4, 1E-1)


    ax.set_ylabel("Frequency $f$ in Hz")
    ax.set_xlabel("Time $t$ in days")
    # ax.colorbar(label=r'Gravitational wave amplitude [1/$\sqrt{\mathrm{Hz}}$]')

    plt.colorbar(pc, label=r'GW amplitude [1/$\sqrt{\mathrm{Hz}}$]')
    # ax.set_title("Spectrogram of data set nr. {:04}".format(n), y = 1.02)


    ax.grid(False)

    plt.savefig("./thesis_plots/plots/chapter_6/{}_spec.png".format(names[i]))





