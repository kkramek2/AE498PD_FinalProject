from grss import fit, prop, utils
import numpy as np
from astropy.time import Time
import matplotlib.pyplot as plt

init_sol = {
    't': 2460522.5 - 2400000.5,  # JD -> MJD
    'e': 0.3906575, 
    'q': 1.00537975, 
    'tp': 2460441.09818 - 2400000.5,  # JD - MJD
    'om': np.deg2rad(214.42379), 
    'w': np.deg2rad(359.96368), 
    'i': np.deg2rad(10.68883),
    }
init_cov = np.diag([
    #0**2,
    (2.1e-5)**2,
    (1.98e-6)**2,
    (0.000675)**2,
    (np.deg2rad(0.0006))**2,
    (np.deg2rad(0.0008))**2,
    (np.deg2rad(0.00027))**2
    ])
optical_obs_file = '2024pdc25.xml'

# load the optical observations
obs_df = fit.get_optical_obs(body_id='2024pdc25', optical_obs_file=optical_obs_file)

# initialize the OD simulation and run the filter
n_iter_max = 10
fit_sim = fit.FitSimulation(init_sol, obs_df, init_cov, n_iter_max=n_iter_max)
fit_sim.filter_lsq()

# print the summary
fit_sim.print_summary()
fit_sim.plot_summary()
fit_sim.iters[-1].plot_iteration_summary(title='Postfit Residuals')

# print refined initial orbit parameters
mean_0 = np.array(list(init_sol.values())[1:])
cov_0 = init_cov
mean_f = np.array(list(fit_sim.x_nom.values()))
cov_f = fit_sim.covariance

print('')
print('Unrefined Initial Orbit Parameters:\n')
print(f'mean_0 = {mean_0}\n')
print(f'cov_0 = {cov_0}\n')
print('')
print('Refined Initial Orbit Parameters:\n')
print(f'mean_f = {mean_f}\n')
print(f'cov_f = {cov_f}\n')
print('')

# define the state of the asteroid
asteroid = prop.IntegBody(
    name='2024pdc25',
    t0=2460522.5 - 2400000.5,  # JD -> MJD
    mass=0.0,
    radius=0.0,
    cometaryState=mean_f,
    ngParams=prop.NongravParameters()
)

# define the final time, initialize the simulation, and set the integration parameters
tf = 66639   # April 30, 2041 --- six days post-impact
prop_sim = prop.PropSimulation(name='2024pdc25', t0=asteroid.t0, defaultSpiceBodies=441, DEkernelPath=utils.default_kernel_path)
prop_sim.set_integration_parameters(tf)
prop_sim.add_integ_body(asteroid)
prop_sim.integrate()

# get sigma-point representation of the refined initial orbit parameters for parallelized Monte Carlo-style uncertainty propagation of asteroid orbit
sigma_points_f = prop.SigmaPoints({"t": fit_sim.t_sol, **fit_sim.x_nom}, cov_f, "merwe", 1e-3, 2.0, 0.0)

# propagate the simulation for each sigma point in parallel
close_approaches, impact_events = prop.parallel_propagate(
    ref_sol={"t": fit_sim.t_sol, **fit_sim.x_nom},
    ref_nongrav=prop.NongravParameters(),
    ref_sim=prop_sim,
    clones=sigma_points_f.sigma_points_dict,
    num_threads=8,
    reconstruct=True
)

# determine the likely encounter / impact time near April 24, 2041
close_approaches_list = [x for sub in close_approaches for x in sub]
impact_events_list = [x for sub in impact_events for x in sub]

earth_close_approaches_list = [x for x in (close_approaches_list) if x.centralBodySpiceId == 399]   # filter for Earth events
earth_impact_events_list = [x for x in (impact_events_list) if x.centralBodySpiceId == 399]         # filter for Earth events
print('')
print("Number of Earth Close Approaches:", len(earth_close_approaches_list))
print("Number of Direct Earth Impacts:", len(earth_impact_events_list))
print('')

clusters = prop.cluster_ca_or_impacts(earth_close_approaches_list, max_duration=10, central_body=399)   # cluster Earth close approaches within a duration of days
earth_cluster = min(clusters, key=lambda c: abs(np.mean([x.t for x in c]) - tf))
cluster_times = np.array([x.t for x in earth_cluster])
mean_t = np.mean(cluster_times)   # mean time of the selected Earth encounter cluster

print("\nEarth Close Approach Times:")
for t in cluster_times:
    print(Time(t, format="mjd", scale="tdb").utc.iso)
print("\nEarth Close Approach Mean Time:", Time(mean_t, format="mjd", scale="tdb").utc.iso)
print('')

# plot the b-plane for the Earth encounter nearest April 24, 2041
prop.plot_bplane(earth_cluster, sigma_points=sigma_points_f, n_std=3.0, scale_coords=False, show_central_body=True, equal_axis=True)
plt.savefig("bplane.png", dpi=300, bbox_inches="tight")
plt.close()