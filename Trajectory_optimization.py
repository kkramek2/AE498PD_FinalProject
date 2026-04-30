import pykep as pk 
import numpy as np
import matplotlib.pyplot as plt
import spiceypy as spice
import pygmo as pg
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
spice.furnsh('/Users/jadendowning/spice_kernels/meta.tm')

#PDC25 orbital elements with epoch 2 data incorporated, refined using GRSS
ref_epoch = 60522.000                  #MJD TDB, solution time for the orbital elements (July 31, 2024)
q = 1.00538004981e+00 * pk.AU          #m
e = 3.90657994905e-01 
a = q / (1 - e)                        #m
i = 1.06888293122e+01 * pk.DEG2RAD     #rad
RAAN = 2.14423769139e+02 * pk.DEG2RAD  #rad
w = 3.59963802784e+02 * pk.DEG2RAD     #rad
t_per = 6.04405982662e+04              #MJD TDB, time of periapse passage

mu_S = 1.32712440018e+20          #m^3/s^2
n = np.sqrt(mu_S / a**3)               #rad/s
dt = (ref_epoch - t_per) * pk.DAY2SEC      #s, delta t to be used in Kepler's law solution
M = n * dt                             #rad, mean anomaly
print('mean anomaly at epoch:', M)
print(i, RAAN, w)

state = spice.conics([q, e, i, RAAN, w, M, ref_epoch, mu_S], ref_epoch) 
r0 = state[:3]                         #m
v0 = state[3:]                         #m/s

#Adding PDC25 as a user defined planet
class PDC25_udpla:
    def __init__(self, ref_epoch, r0, v0, mu_S, name='PDC25'):
        self.ref_epoch = ref_epoch
        self.r0 = r0
        self.v0 = v0
        self.mu = mu_S
        self.mu_central_body = mu_S
        self.name = name

    def eph(self, mjd2000):
        dt = (mjd2000.mjd2000 - self.ref_epoch) * pk.DAY2SEC
        r, v = pk.propagate_lagrangian(self.r0, self.v0, dt, self.mu)
        return r, v
    
    def get_name(self):
        return self.name
    
    def compute_period(self, t0):
        r = np.linalg.norm(self.r0)
        v = np.linalg.norm(self.v0)
        E = (v**2)/2 - self.mu / r
        a = -self.mu / (2*E)
        T = 2 * np.pi * np.sqrt(a**3 / self.mu)
        return T


PDC25 = PDC25_udpla(ref_epoch, r0, v0, mu_S, name='PDC25')

#Define Earth and PDC25 as start and target bodies
start = pk.planet.jpl_lp('earth')
target = PDC25

#Set up the multi-impulse solver
t0_min = (spice.str2et('2029-04-15T00:00:00.000') / pk.DAY2SEC)     #MJD2000, minimum departure time
t0_max = (spice.str2et('2029-10-10T00:00:00.000') / pk.DAY2SEC)     #MJD2000, maximum departure time
print(t0_min, t0_max)

tof_bounds = [350, 520]                    #days, bounds for time of flight
DV_max_bounds = [0, 1000.0]                #m/s, bounds for delta-V
phase_free = False
multi_objective = False                                   #CURRENTLY ONLY OPTIMIZING DV, TRY WITH TOF LATER
t0_bounds = [t0_min, t0_max]

udp3 = pk.trajopt.pl2pl_N_impulses(start=start, target=target, N_max=3, phase_free=phase_free, multi_objective=multi_objective, t0=t0_bounds, tof=tof_bounds, vinf=DV_max_bounds)

def solve(udp, N=20):
    uda = pg.cmaes(4500, force_bounds=True, sigma0=0.5, ftol=1e-4)
    algo = pg.algorithm(uda)

    print("Multi-start:")

    res = list()
    for i in range(N):
        pop = pg.population(udp, 20)
        pop = algo.evolve(pop)
        res.append([pop.champion_f, pop.champion_x])
        print(i, pop.champion_f[0], end= '\r')
        
    best_x = sorted(res, key =  lambda x: x[0][0])[0][1]
    print(f"\nThe best solution found has a DV of {udp.fitness(best_x)[0]/1000:.5e} km/s")

    return best_x

best_x_3 = solve(udp3)

udp3.pretty(best_x_3)

def visualise(udp, best_x):
    # Create the figure and define the grid
    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(
        2, 2, width_ratios=[2, 1], height_ratios=[1, 1], wspace=0.01, hspace=0.01)

    # Top view (spans rows and columns)
    ax1 = fig.add_subplot(projection="3d")
    udp.plot(best_x, axes=ax1)
    ax1.view_init(90, 0)
    ax1.axis("off")

    #Ecliptic view 1
    ax2 = fig.add_subplot(gs[0, 1], projection="3d")
    udp.plot(best_x, axes=ax2)
    ax2.view_init(0, 0)
    ax2.axis("off")

   # Ecliptic view 2
    ax3 = fig.add_subplot(gs[1, 1], projection="3d")
    udp.plot(best_x, axes=ax3)
    ax3.view_init(0, 90)
    ax3.axis("off")

    ax1.set_xlabel('x (au)', fontsize=12)
    ax1.set_ylabel('y (au)', fontsize=12)
    ax1.set_zlabel('z (au)', fontsize=12)
    ax1.set_title('Transfer Trajectory from Earth to PDC25', fontsize=14)

    return fig

fig = visualise(udp3, best_x_3)

sun = mlines.Line2D([], [], color='y', marker='o', linestyle='None', label='Sun')
earth = mlines.Line2D([], [], color=(0.6, 0.54, 0.6), label='Earth Orbit at Departure')
target = mlines.Line2D([], [], color=(0.9, 0.7, 0.9), label='PDC25 Orbit at Impact')
fig.legend(handles=[sun, earth, target], loc='upper right', bbox_to_anchor=(0.9, 0.87), fontsize=12)
plt.show()

