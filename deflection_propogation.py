import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton, brentq
from scipy.integrate import solve_ivp
import os
from grss import prop, utils

## HELPER FUNCTIONS ##

# convert AU to km
def au2km(au):
    AU_IN_KM = 149_597_870.7
    return np.asarray(au) * AU_IN_KM

# convert km to au
def km2au(km):
    AU_IN_KM = 149_597_870.7
    return np.asarray(km) / AU_IN_KM

# convert AU/day to km/s
def au_per_day2km_per_s(au_per_day):
    AU_IN_KM = 149_597_870.7
    DAY_IN_S = 86400.0
    return np.asarray(au_per_day) * AU_IN_KM / DAY_IN_S

# convert km/s to AU/day
def km_per_s2au_per_day(km_per_s):
    AU_IN_KM = 149_597_870.7
    DAY_IN_S = 86400.0
    return np.asarray(km_per_s) * DAY_IN_S / AU_IN_KM

# convert AU^3/day^2 to km^3/s^2
def au3_per_day2km3_per_s2(au3_per_day2):
    AU_IN_KM = 149_597_870.7
    DAY_IN_S = 86400.0
    return np.asarray(au3_per_day2) * AU_IN_KM**3 / DAY_IN_S**2

# get semi major axis a
def get_semimajor_axis(q, e): return q / (1 - e)

# wrap angles between [0, 2pi)
def wrap_angle(angle): return angle % (2.0 * np.pi)

# convert GRSS cometary state to the order expected by prop.IntegBody
def grss_cometary_state(sol): return [float(sol["e"]), float(sol["q"]), float(sol["tp"]), float(sol["om"]), float(sol["w"]), float(sol["i"])]

# function to compute the local orbital basis (rhat, that, hhat) from an instantaneous heliocentric state
def get_rtn_basis(r, v):
    rhat = r / np.linalg.norm(r)
    hvec = np.cross(r, v)
    hhat = hvec / np.linalg.norm(hvec)
    that = np.cross(hhat, rhat)
    return rhat, that, hhat

## MAIN FUNCTIONS ##

# solve for eccentric anomaly
def compute_E(M, e):
    M = wrap_angle(M)

    # define keplers equation and its derivative
    def kepler(E): return E - e*np.sin(E) - M
    def kepler_prime(E): return 1 - e*np.cos(E)

    # solve for E
    E = float(newton(func=kepler, x0=M, fprime=kepler_prime))
    return E

# plot the b-plane visualizations
def plot_bplane(xi, zeta, b_coll, R, title="Opik b-plane Plot", save=True, units_in_km=False):
    theta = np.linspace(0, 2*np.pi, 800)
    fig, ax = plt.subplots(figsize=(11,11))
    ax.plot(R*np.cos(theta), R*np.sin(theta), label="Earth Radius")
    ax.plot(4*R*np.cos(theta), 4*R*np.sin(theta), label="4x Earth Radius")
    ax.plot(b_coll*np.cos(theta), b_coll*np.sin(theta), label="Collision Radius")
    ax.scatter([xi], [zeta], marker="x", s=80, label="Encounter Point (xi, zeta)")
    max_extent = max(4*R, abs(xi), abs(zeta))
    pad = 1.1 * max_extent
    ax.set_xlim(-pad, pad)
    ax.set_ylim(-pad, pad)
    ax.set_aspect('equal', adjustable='box')
    unit_label = "km" if units_in_km else "AU"
    ax.set_xlabel(rf"$\xi$ ({unit_label})")
    ax.set_ylabel(rf"$\zeta$ ({unit_label})")
    ax.set_title(title)
    ax.grid(True)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    if save:
        filename = f"{title}.png"
        fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

# function to compute b-plane quantities directly from current states
def get_bplane_now(r1, v1, r2, v2, mu2, R_planet):

    # get the relative state
    r_rel, v_rel = r2 - r1, v2 - v1

    # compute S_hat
    v_inf = np.linalg.norm(v_rel)
    S_hat = v_rel / v_inf

    # compute T_hat
    T_hat = np.cross(v1, S_hat) / np.linalg.norm(np.cross(v1, S_hat))

    # compute R_hat
    R_hat = np.cross(S_hat, T_hat)

    # compute B
    h = np.cross(r_rel, v_rel)
    B = np.cross(S_hat, h) / v_inf

    # compute collision radius
    b_coll = R_planet * np.sqrt(1.0 + 2.0 * mu2 / (R_planet * v_inf**2))

    # compute xi and zeta
    xi = float(np.dot(B, T_hat))
    zeta = float(np.dot(B, R_hat))

    return xi, zeta, B, b_coll, S_hat, T_hat, R_hat, v_inf

# build Valsecchi circles for a given b-plane state (derived from https://github.com/eggls6/bplane/blob/main/opik_apophis_bplane_keyholes.ipynb)
def compute_valsecchi_circles(rE, vE, rA, vA, mu1, mu2, R_planet, elements1, units_in_km=False):

    # get the relative state
    r_rel = rA - rE
    v_rel = vA - vE

    # compute S_hat
    v_inf = np.linalg.norm(v_rel)
    S_hat = v_rel / v_inf

    # compute T_hat
    T_hat = np.cross(vE, S_hat)
    T_hat = T_hat / np.linalg.norm(T_hat)

    # compute R_hat
    R_hat = np.cross(S_hat, T_hat)

    # compute B
    h = np.cross(r_rel, v_rel)
    B = np.cross(S_hat, h) / v_inf

    # compute collision radius
    b_coll = R_planet * np.sqrt(1.0 + 2.0*mu2/(R_planet * v_inf**2))

    # compute xi and zeta
    xi = float(np.dot(B, T_hat))
    zeta = float(np.dot(B, R_hat))

    # convert to km and km/s (and mu_sun to km^3/s^2)
    xi = au2km(xi); zeta = au2km(zeta); b_coll = au2km(b_coll); v_inf = au_per_day2km_per_s(v_inf); rE = au2km(rE); vE = au_per_day2km_per_s(vE); mu1 = au3_per_day2km3_per_s2(mu1); mu2 = au3_per_day2km3_per_s2(mu2)

    # local heliocentric encounter frame for Valsecchi angles (X,Y,Z)
    r_hat = rE / np.linalg.norm(rE)
    v_hat_raw = vE / np.linalg.norm(vE)
    y_hat = v_hat_raw - np.dot(v_hat_raw, r_hat)*r_hat
    y_hat = y_hat / np.linalg.norm(y_hat)
    z_hat = np.cross(r_hat, y_hat)
    z_hat = z_hat / np.linalg.norm(z_hat)
    x_hat = r_hat

    # incoming asymptote components in this frame
    u_inf_minus = -v_inf * np.array(S_hat, dtype=float)
    Ux = float(np.dot(u_inf_minus, x_hat))
    Uy = float(np.dot(u_inf_minus, y_hat))
    Uz = float(np.dot(u_inf_minus, z_hat))
    U  = float(np.linalg.norm(u_inf_minus))
    theta = float(np.arccos(np.clip(Uy / U, -1.0, 1.0)))
    phi   = float(np.arctan2(Ux, Uz))
    c = float(mu2 / (v_inf**2))

    # compute theta* for resonance using local circular approx (directly from https://github.com/eggls6/bplane/blob/main/opik_apophis_bplane_keyholes.ipynb)
    qE, eE, *_ = elements1
    aE = get_semimajor_axis(qE, eE)
    aE = float(au2km(aE))
    rE = float(np.linalg.norm(rE))
    v_c = float(np.sqrt(mu1 / rE))
    U_ratio = float(v_inf / v_c)

    def theta_prime_for_resonance(p, q):
        a_res = aE * (p / q)**(2/3)
        v_res2 = mu1 * (2.0 / rE - 1.0 / a_res)
        cos_thp = (v_res2 / (v_c*v_c) - 1.0 - U_ratio*U_ratio) / (2.0 * U_ratio)
        return a_res, cos_thp
    
    def valsecchi_circle_params(theta, theta_p, c):
        denom = (np.cos(theta_p) - np.cos(theta))
        D = c * np.sin(theta)   / denom
        R = c * np.sin(theta_p) / denom
        return D, R
    
    res_list = [(7,6),(6,5),(5,4),(4,3),(3,2),(2,1)]
    circle_data = []
    for (p, q) in res_list:
        a_res, cos_thp = theta_prime_for_resonance(p, q)
        if abs(cos_thp) <= 1.0:
            thp = float(np.arccos(cos_thp))
            D, R = valsecchi_circle_params(theta, thp, c)
            if units_in_km:
                circle_data.append((p, q, a_res, thp, D, R))
            else:
                circle_data.append((p, q, km2au(a_res), thp, km2au(D), km2au(R)))

    # convert back to AU
    if not units_in_km:
        xi = km2au(xi); zeta = km2au(zeta); b_coll = km2au(b_coll) 

    return xi, zeta, b_coll, circle_data, theta, phi

# plot the b-plane visualization with Valsecchi circles
def plot_bplane_valsecchi(xi, zeta, b_coll, R, circle_data, title="Impact b-plane w/ Valsecchi Circles", save=True, units_in_km=False):
    theta = np.linspace(0, 2*np.pi, 800)
    fig, ax = plt.subplots(figsize=(11,11))
    ax.plot(R*np.cos(theta), R*np.sin(theta), label="Earth Radius")
    ax.plot(4*R*np.cos(theta), 4*R*np.sin(theta), label="4x Earth Radius")
    ax.plot(b_coll*np.cos(theta), b_coll*np.sin(theta), label="Collision Radius")
    for (p, q, a_res, thp, D, Rc) in circle_data:
        xi_c   = Rc * np.sin(theta)
        zeta_c = D  + Rc * np.cos(theta)
        ax.plot(xi_c, zeta_c, linewidth=1.5, label=f"Valsecchi {p}:{q}")
    ax.scatter([xi], [zeta], marker="x", s=80, label="Encounter Point (xi, zeta)")
    max_extent = max(4*R, abs(xi), abs(zeta))
    for (_, _, _, _, D, Rc) in circle_data: max_extent = max(max_extent, abs(D) + abs(Rc))
    pad = 1.1 * max_extent
    ax.set_xlim(-pad, pad)
    ax.set_ylim(-pad, pad)
    ax.set_aspect('equal', adjustable='box')
    unit_label = "km" if units_in_km else "AU"
    ax.set_xlabel(rf"$\xi$ ({unit_label})")
    ax.set_ylabel(rf"$\zeta$ ({unit_label})")
    ax.set_title(title)
    ax.grid(True)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    if save:
        filename = f"{title}.png"
        fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    
# function to initialize a GRSS sim from a GRSS solution dict for a secondary body
def init_grss_sim(sol_secondary, tf_abs, body_name="secondary"):
    dekernel_dir = str(utils.default_kernel_path)
    if not os.path.isdir(dekernel_dir):
        raise FileNotFoundError(f"GRSS kernel directory not found: {dekernel_dir}")
    expected = os.path.join(dekernel_dir, "de441.bsp")
    if not os.path.isfile(expected):
        raise FileNotFoundError(f"Expected DE441 kernel not found: {expected}")
    asteroid = prop.IntegBody(name=body_name, t0=float(sol_secondary["t"]), mass=0.0, radius=0.0, cometaryState=grss_cometary_state(sol_secondary), ngParams=prop.NongravParameters())
    sim = prop.PropSimulation(name=body_name, t0=float(sol_secondary["t"]), defaultSpiceBodies=441, DEkernelPath=dekernel_dir)
    sim.map_ephemeris()
    sim.add_integ_body(asteroid)
    sim.set_integration_parameters(float(tf_abs))
    return sim

# function to build and integrate a GRSS sim from a GRSS solution dict for a secondary body
def build_grss_sim(sol_secondary, tf_abs, body_name="secondary"):
    sim = init_grss_sim(sol_secondary, tf_abs, body_name=body_name)
    sim.integrate()
    sim.map_ephemeris()
    return sim

# function to get the states of the main and secondary bodies at a given absolute time from a GRSS sim
def get_states_grss(sim, t_abs, main_body_name="Earth"):
    sim.map_ephemeris()
    x_main = np.asarray(sim.get_spiceBody_state(float(t_abs), main_body_name), dtype=float).reshape(-1)
    x_secondary = np.asarray(sim.interpolate(float(t_abs)), dtype=float).reshape(-1)
    if x_secondary.size < 6:
        raise ValueError(f"Expected at least 6 state values, got shape {x_secondary.shape}")
    x_secondary = x_secondary[:6]
    r_main = x_main[:3]
    v_main = x_main[3:6]
    r_secondary = x_secondary[:3]
    v_secondary = x_secondary[3:6]
    return r_main, v_main, r_secondary, v_secondary

# function to compute the relative distance between the main and secondary bodies at a given absolute time from a GRSS sim
def get_relative_distance_grss(sim, t_abs, main_body_name="Earth"):
    r_main, _, r_secondary, _ = get_states_grss(sim, t_abs, main_body_name=main_body_name)
    return float(np.linalg.norm(r_secondary - r_main))

# function to refine the SOF crossing time by searching for the minimum relative distance between the main and secondary bodies in a given time window around the initial SOF crossing time
def refine_close_approach_time_grss(sim, t_lo, t_hi, n_fine=1001, main_body_name="Earth"):
    times = np.linspace(float(t_lo), float(t_hi), int(n_fine))
    dists = np.array([get_relative_distance_grss(sim, t, main_body_name=main_body_name) for t in times], dtype=float)
    i_min = int(np.argmin(dists))
    return float(times[i_min]), float(dists[i_min])

# function to find all SOF encounters (entry and exit times, closest approach time and distance, b-plane geometry) in a given time window from a GRSS sim of the nominal orbit
def find_all_sof_encounters_grss(sim, mu_main, R_main, R_sof, t_search_start_abs, tf_abs, dt_days, main_body_name="Earth"):
    def dist_to_sof(t_abs): return get_relative_distance_grss(sim, t_abs, main_body_name=main_body_name) - R_sof

    # initialize search variables
    encounters = []
    t_prev = float(t_search_start_abs)
    d_prev = dist_to_sof(t_prev)
    inside = (d_prev <= 0.0)
    if inside:
        t_entry = t_prev
        best_t = t_prev
        best_d = get_relative_distance_grss(sim, t_prev, main_body_name=main_body_name)
    else:
        t_entry = None
        best_t = None
        best_d = None

    # search forward in time for SOF crossings and record encounter data
    while t_prev < tf_abs:
        t_next = min(t_prev + dt_days, tf_abs)
        d_curr = dist_to_sof(t_next)
        if (not inside) and (d_prev > 0.0) and (d_curr <= 0.0):
            t_entry = float(brentq(dist_to_sof, t_prev, t_next))
            inside = True
            best_t = t_next
            best_d = get_relative_distance_grss(sim, t_next, main_body_name=main_body_name)
        if inside:
            d_now = get_relative_distance_grss(sim, t_next, main_body_name=main_body_name)
            if (best_d is None) or (d_now < best_d):
                best_t = t_next
                best_d = d_now
        if inside and (d_prev <= 0.0) and (d_curr > 0.0):
            t_exit = float(brentq(dist_to_sof, t_prev, t_next))
            t_ca_lo = max(t_entry, best_t - dt_days)
            t_ca_hi = min(t_exit, best_t + dt_days)
            t_ca, d_ca = refine_close_approach_time_grss(sim, t_ca_lo, t_ca_hi, main_body_name=main_body_name)
            r_main, v_main, r_secondary, v_secondary = get_states_grss(sim, t_entry, main_body_name=main_body_name)
            xi, zeta, B, b_coll, S_hat, T_hat, R_hat, v_inf = get_bplane_now(r_main, v_main, r_secondary, v_secondary, mu_main, R_main)
            encounters.append({
                "t_entry_abs": float(t_entry),
                "t_entry_rel": float(t_entry - t_search_start_abs),
                "t_exit_abs": float(t_exit),
                "t_exit_rel": float(t_exit - t_search_start_abs),
                "t_ca_abs": float(t_ca),
                "t_ca_rel": float(t_ca - t_search_start_abs),
                "d_ca": float(d_ca),
                "xi": float(xi),
                "zeta": float(zeta),
                "B": B,
                "b": float(np.linalg.norm(B)),
                "b_coll": float(b_coll),
                "S_hat": S_hat,
                "T_hat": T_hat,
                "R_hat": R_hat,
                "v_inf": float(v_inf),
            })
            inside = False
            t_entry = None
            best_t = None
            best_d = None
        t_prev = t_next
        d_prev = d_curr

    # if ended inside the SOF, record an encounter with exit time at the end of the search window
    if inside:
        t_exit = float(tf_abs)
        t_ca_lo = max(t_entry, best_t - dt_days)
        t_ca_hi = min(t_exit, best_t + dt_days)
        t_ca, d_ca = refine_close_approach_time_grss(sim, t_ca_lo, t_ca_hi, main_body_name=main_body_name)
        r_main, v_main, r_secondary, v_secondary = get_states_grss(sim, t_entry, main_body_name=main_body_name)
        xi, zeta, B, b_coll, S_hat, T_hat, R_hat, v_inf = get_bplane_now(r_main, v_main, r_secondary, v_secondary, mu_main, R_main)
        encounters.append({
            "t_entry_abs": float(t_entry),
            "t_entry_rel": float(t_entry - t_search_start_abs),
            "t_exit_abs": float(t_exit),
            "t_exit_rel": float(t_exit - t_search_start_abs),
            "t_ca_abs": float(t_ca),
            "t_ca_rel": float(t_ca - t_search_start_abs),
            "d_ca": float(d_ca),
            "xi": float(xi),
            "zeta": float(zeta),
            "B": B,
            "b": float(np.linalg.norm(B)),
            "b_coll": float(b_coll),
            "S_hat": S_hat,
            "T_hat": T_hat,
            "R_hat": R_hat,
            "v_inf": float(v_inf),
        })

    return encounters

# function to find the first absolute MJD TDB time at which the secondary enters the SOF sphere
def find_sof_crossing_time_grss(sim, R_sof, t_search_start_abs, tf_abs, dt_days, main_body_name="Earth"):
    def dist_to_sof(t_abs):
        r_main, _, r_secondary, _ = get_states_grss(sim, t_abs, main_body_name=main_body_name)
        return float(np.linalg.norm(r_secondary - r_main) - R_sof)
    t_prev = float(t_search_start_abs)
    d_prev = dist_to_sof(t_prev)
    if d_prev <= 0.0:
        return t_prev
    while t_prev < tf_abs:
        t_next = min(t_prev + dt_days, tf_abs)
        d_curr = dist_to_sof(t_next)
        if (d_prev > 0.0) and (d_curr <= 0.0):
            return float(brentq(dist_to_sof, t_prev, t_next))
        if t_next == tf_abs:
            break
        t_prev = t_next
        d_prev = d_curr
    return None

# function to propagate the asteroid covariance cloud under a fixed event schedule
def evaluate_bplane_cloud_grss(sol_secondary, cov_secondary, events, mu_main, R_main, R_sof, t_max, dt, t_search_start_abs=None, robust_metric="min_b", main_body_name="Earth", body_name="secondary"):
    t0_abs = float(sol_secondary["t"])
    tf_abs = t0_abs + float(t_max)
    if t_search_start_abs is None:
        t_search_start_abs = t0_abs

    # create the sigma points for the secondary body's state covariance
    sigma_points = prop.SigmaPoints(sol_secondary, cov_secondary, "merwe", 1e-3, 2.0, 0.0)

    # propagate each clone with the given events and compute the b-plane geometry at SOF crossing
    clones = sigma_points.sigma_points_dict
    if hasattr(clones, "values"): clone_iter = clones.values()
    else: clone_iter = clones
    clone_results = []
    for clone in clone_iter:
        clone_sol = clone_sol_dict(clone)
        if clone_sol["t"] is None:
            clone_sol["t"] = t0_abs

        # build and integrate the GRSS sim for this clone with the given events
        sim = init_grss_sim(clone_sol, tf_abs, body_name=body_name)
        for evt in events:
            sim.add_event(evt)
        sim.integrate()

        # find the SOF crossing time and compute the b-plane geometry at that time
        t_sof_abs = find_sof_crossing_time_grss(sim=sim, R_sof=R_sof, t_search_start_abs=t_search_start_abs, tf_abs=tf_abs, dt_days=dt, main_body_name=main_body_name)
        if t_sof_abs is None:
            clone_results.append(None)
            continue

        # get the states at the SOF crossing time and compute the b-plane geometry
        r_main, v_main, r_secondary, v_secondary = get_states_grss(sim, t_sof_abs, main_body_name=main_body_name)
        xi, zeta, B, b_coll, S_hat, T_hat, R_hat, v_inf = get_bplane_now(r_main, v_main, r_secondary, v_secondary, mu_main, R_main)

        # append results to clone
        clone_results.append({
            "sim": sim,
            "t_sof_abs": float(t_sof_abs),
            "t_sof_rel": float(t_sof_abs - t0_abs),
            "xi": float(xi),
            "zeta": float(zeta),
            "B": B,
            "b": float(np.linalg.norm(B)),
            "b_coll": float(b_coll),
            "S_hat": S_hat,
            "T_hat": T_hat,
            "R_hat": R_hat,
            "v_inf": float(v_inf),
        })

    # 
    feasible = [r for r in clone_results if r is not None]
    if not feasible:
        return None
    b_vals = np.array([r["b"] for r in feasible], dtype=float)

    # 
    if robust_metric == "min_b":
        b_robust = float(np.min(b_vals))
    elif robust_metric == "mean_b":
        b_robust = float(np.mean(b_vals))
    elif robust_metric == "p10_b":
        b_robust = float(np.percentile(b_vals, 10.0))
    else:
        raise ValueError("robust_metric must be 'min_b', 'mean_b', or 'p10_b'")

    # 
    worst_idx = int(np.argmin(b_vals))
    worst_case = feasible[worst_idx]

    return {
        "sigma_points": sigma_points,
        "clone_results": feasible,
        "b_vals": b_vals,
        "b_robust": b_robust,
        "worst_case": worst_case,
    }

# function to get the nominal SOF entry time and b-plane geometry from a GRSS sim of the nominal orbit
def get_nominal_sof_and_bplane_grss(sol_secondary, mu_main, R_main, R_sof, t_max, dt, cov_secondary=None, robust_metric="min_b", main_body_name="Earth", body_name="secondary"):
    t0_abs = float(sol_secondary["t"])
    tf_abs = t0_abs + float(t_max)
    sim = build_grss_sim(sol_secondary, tf_abs, body_name=body_name)
    encounters = find_all_sof_encounters_grss(sim=sim, mu_main=mu_main, R_main=R_main, R_sof=R_sof, t_search_start_abs=t0_abs, tf_abs=tf_abs, dt_days=dt, main_body_name=main_body_name)
    if len(encounters) == 0: return None
    return {
        "sim": sim,
        "encounters": encounters,
    }

# function to change a sigma-point clone into the GRSS solution-dict format
def clone_sol_dict(clone):
    if isinstance(clone, dict):
        return {
            "t": float(clone["t"]),
            "e": float(clone["e"]),
            "q": float(clone["q"]),
            "tp": float(clone["tp"]),
            "om": float(clone["om"]),
            "w": float(clone["w"]),
            "i": float(clone["i"]),
        }

    # otherwise assume it's an array-like in the order [e, q, tp, om, w, i] and convert to dict
    arr = np.asarray(clone, dtype=float).reshape(-1)
    if arr.size != 6:
        raise ValueError(f"Expected clone of length 6 or GRSS solution dict, got shape {arr.shape}")
    
    return {
        "t": None,
        "e": float(arr[0]),
        "q": float(arr[1]),
        "tp": float(arr[2]),
        "om": float(arr[3]),
        "w": float(arr[4]),
        "i": float(arr[5]),
    }

# function to build a GRSS event for applying an impulse in either the radial, transverse, or normal direction
def build_impulse_event_from_nominal(sim_nominal, t_impulse_abs, dV_signed, direction, body_name="secondary", main_body_name="Earth"):
    _, _, r_secondary, v_secondary = get_states_grss(sim_nominal, t_impulse_abs, main_body_name=main_body_name)
    rhat, that, hhat = get_rtn_basis(r_secondary, v_secondary)

    # determine the applied direction
    if direction == "radial":
        uhat = rhat
    elif direction == "transverse":
        uhat = that
    elif direction == "normal":
        uhat = hhat
    else:
        raise ValueError("direction must be 'radial', 'transverse', or 'normal'")

    # build the GRSS event
    evt = prop.Event()
    evt.bodyName = body_name
    evt.t = float(t_impulse_abs)
    evt.isContinuous = False
    evt.deltaV = list(float(dV_signed) * uhat)
    evt.multiplier = 1.0
    return evt

# function to solve for the minimum required impulse at a given time to achieve a desired b-plane miss distance at SOF crossing
def solve4accel2perturb_instantaneous_at_time_grss(sol_secondary, cov_secondary, mu_main, R_main, R_sof, direction, b_target, t_impulse, t_max=365.0, dt=1.0, robust_metric="min_b", main_body_name="Earth", body_name="secondary"):
    t0_abs = float(sol_secondary["t"])
    tf_abs = t0_abs + float(t_max)
    t_impulse_abs = t0_abs + float(t_impulse)

    # build the nominal sim to use for generating the impulse event
    sim_nominal = build_grss_sim(sol_secondary, tf_abs, body_name=body_name)

    # define a function to compute the b-plane geometry at SOF crossing for a given impulse magnitude
    def cloud_at_impulse(dV_signed):
        evt = build_impulse_event_from_nominal(sim_nominal=sim_nominal, t_impulse_abs=t_impulse_abs, dV_signed=dV_signed, direction=direction, body_name=body_name, main_body_name=main_body_name)
        cloud = evaluate_bplane_cloud_grss(sol_secondary=sol_secondary, cov_secondary=cov_secondary, events=[evt], mu_main=mu_main, R_main=R_main, R_sof=R_sof, t_max=t_max, dt=dt, t_search_start_abs=t_impulse_abs, robust_metric=robust_metric, main_body_name=main_body_name, body_name=body_name)
        if cloud is None: return None, None
        return cloud["b_robust"], cloud

    # define a function to compute the difference between the robust b-plane metric and the target for a given impulse magnitude
    def comp_b_diff(dV_signed):
        b_robust, _ = cloud_at_impulse(dV_signed)
        if b_robust is None:
            return np.nan
        return b_robust - b_target

    # check if zero impulse already achieves the target b-plane miss distance
    f0 = comp_b_diff(0.0)
    if not np.isnan(f0) and f0 >= 0.0:
        b0, cloud0 = cloud_at_impulse(0.0)
        return 0.0, b0, cloud0["worst_case"]

    # use a root-finding method to solve for the required impulse
    dV_hi = 1e-12
    f_plus = comp_b_diff(+dV_hi)
    f_minus = comp_b_diff(-dV_hi)
    if np.isnan(f_plus) and np.isnan(f_minus):
        raise RuntimeError("Neither sign yields an SOF crossing at tested deltaV.")
    elif np.isnan(f_plus):
        sgn = -1.0
    elif np.isnan(f_minus):
        sgn = +1.0
    else:
        sgn = +1.0 if f_plus > f_minus else -1.0

    # exponentially expand the search bounds until a sign change is found in the b-plane difference function
    for _ in range(80):
        f = comp_b_diff(sgn * dV_hi)
        if np.isnan(f):
            dV_hi *= 2.0
            continue
        if f >= 0.0:
            break
        dV_hi *= 2.0
    else:
        raise RuntimeError("Failed to find brackets for the required deltaV.")

    # use a root-finding method to solve for the required impulse that achieves the target b-plane miss distance
    dV_min = sgn * float(brentq(lambda x: comp_b_diff(sgn * x), 0.0, dV_hi, xtol=1e-12, rtol=1e-10, maxiter=200))
    b_robust, cloud = cloud_at_impulse(dV_min)
    while b_robust < b_target:
        dV_min *= 1.0000001
        b_robust, cloud = cloud_at_impulse(dV_min)

    return dV_min, b_robust, cloud["worst_case"]

# function to perform a coarse-to-fine search for the best impulse time to achieve a desired b-plane miss distance at SOF crossing with minimum required deltaV
def search_best_impulse_time_grss(sol_secondary, cov_secondary, mu_main, R_main, R_sof, direction, b_target, t_impulse_min, t_impulse_max, t_max=365.0, dt=1.0, coarse_step=5.0, mid_half_width=10.0, mid_step=1.0, fine_half_width=2.0, fine_step=0.1, robust_metric="min_b", main_body_name="Earth", body_name="secondary"):
    best = None

    # define a function to try a grid of impulse times and update the best solution
    def try_time_grid(t_grid):
        nonlocal best
        for t_impulse in t_grid:
            try:
                dV_min, b_val, b_elems = solve4accel2perturb_instantaneous_at_time_grss(sol_secondary=sol_secondary, cov_secondary=cov_secondary, mu_main=mu_main, R_main=R_main, R_sof=R_sof, direction=direction, b_target=b_target, t_impulse=t_impulse, t_max=t_max, dt=dt, robust_metric=robust_metric, main_body_name=main_body_name, body_name=body_name)
            except Exception:
                continue
            if best is None or abs(dV_min) < abs(best["dV_min"]):
                best = {
                    "t_impulse": float(t_impulse),
                    "dV_min": float(dV_min),
                    "b": float(b_val),
                    "b_elems": b_elems,
                }

    # try the coarse grid of impulse times
    coarse_times = np.arange(t_impulse_min, t_impulse_max + 1e-12, coarse_step)
    try_time_grid(coarse_times)
    if best is None:
        raise RuntimeError("No feasible impulse time found in coarse search.")

    # refine around the best coarse solution with a mid grid
    mid_min = max(t_impulse_min, best["t_impulse"] - mid_half_width)
    mid_max = min(t_impulse_max, best["t_impulse"] + mid_half_width)
    mid_times = np.arange(mid_min, mid_max + 1e-12, mid_step)
    try_time_grid(mid_times)

    # refine around the best mid solution with a fine grid
    fine_min = max(t_impulse_min, best["t_impulse"] - fine_half_width)
    fine_max = min(t_impulse_max, best["t_impulse"] + fine_half_width)
    fine_times = np.arange(fine_min, fine_max + 1e-12, fine_step)
    try_time_grid(fine_times)

    return best

# function to propagate the covariance of the secondary body using GRSS and return the close approaches and impact events for the sigma points
def propagate_secondary_covariance_grss(sol_secondary, cov_secondary, mu_main, R_main, R_sof, t_max, dt, events=None, robust_metric="min_b", main_body_name="Earth", body_name="secondary"):
    if events is None: events = []
    return evaluate_bplane_cloud_grss(sol_secondary=sol_secondary, cov_secondary=cov_secondary, events=events, mu_main=mu_main, R_main=R_main, R_sof=R_sof, t_max=t_max, dt=dt, robust_metric=robust_metric, main_body_name=main_body_name, body_name=body_name)

# function to build a list of piecewise continuous thrust events from a nominal GRSS sim
def build_piecewise_continuous_events_from_nominal(sim_nominal, t_start_abs, tf_abs, a_signed, direction, burn_step=0.25, body_name="secondary", main_body_name="Earth"):
    events = []
    t = float(t_start_abs)

    # step forward in time and build events until we reach the end time
    while t < tf_abs - 1e-12:
        dt_seg = min(float(burn_step), tf_abs - t)
        _, _, r_secondary, v_secondary = get_states_grss(sim_nominal, t, main_body_name=main_body_name)
        rhat, that, hhat = get_rtn_basis(r_secondary, v_secondary)

        # determine the applied direction
        if direction == "radial":
            uhat = rhat
        elif direction == "transverse":
            uhat = that
        elif direction == "normal":
            uhat = hhat
        else:
            raise ValueError("direction must be 'radial', 'transverse', or 'normal'")

        # build the GRSS event for this burn segment
        evt = prop.Event()
        evt.bodyName = body_name
        evt.t = float(t)
        evt.isContinuous = False
        evt.deltaV = list(float(a_signed * dt_seg) * uhat)
        evt.multiplier = 1.0
        events.append(evt)

        # step forward in time for the next segment
        t += dt_seg

    return events

# function to solve for the required continuous deltaV starting at a given time to achieve a desired b-plane miss distance at SOF crossing
def solve4accel2perturb_continuous_at_time_grss(sol_secondary, cov_secondary, mu_main, R_main, R_sof, direction, b_target, t_start, t_max=365.0, dt=1.0, burn_step=0.25, robust_metric="min_b", main_body_name="Earth", body_name="secondary"):
    t0_abs = float(sol_secondary["t"])
    tf_abs = t0_abs + float(t_max)
    t_start_abs = t0_abs + float(t_start)

    # build the nominal sim to use for generating the piecewise continuous thrust events
    sim_nominal = build_grss_sim(sol_secondary, tf_abs, body_name=body_name)

    # define a function to compute the b-plane geometry at SOF crossing for a given continuous acceleration magnitude
    def cloud_at_piecewise(a_signed):
        events = build_piecewise_continuous_events_from_nominal(sim_nominal=sim_nominal, t_start_abs=t_start_abs, tf_abs=tf_abs, a_signed=a_signed, direction=direction, burn_step=burn_step, body_name=body_name, main_body_name=main_body_name)
        cloud = evaluate_bplane_cloud_grss(sol_secondary=sol_secondary, cov_secondary=cov_secondary, events=events, mu_main=mu_main, R_main=R_main, R_sof=R_sof, t_max=t_max, dt=dt, t_search_start_abs=t_start_abs, robust_metric=robust_metric, main_body_name=main_body_name, body_name=body_name)
        if cloud is None:
            return None, None
        return cloud["b_robust"], cloud

    # define a function to compute the difference between the robust b-plane metric and the target for a given continuous acceleration magnitude
    def comp_b_diff(a_signed):
        b_robust, _ = cloud_at_piecewise(a_signed)
        if b_robust is None:
            return np.nan
        return b_robust - b_target

    # check if zero acceleration already achieves the target b-plane miss distance
    f0 = comp_b_diff(0.0)
    if not np.isnan(f0) and f0 >= 0.0:
        b0, cloud0 = cloud_at_piecewise(0.0)
        return 0.0, 0.0, b0, cloud0["worst_case"]

    # use a root-finding method to solve for the required continuous acceleration
    a_hi = 1e-14
    f_plus = comp_b_diff(+a_hi)
    f_minus = comp_b_diff(-a_hi)
    if np.isnan(f_plus) and np.isnan(f_minus):
        raise RuntimeError("Neither sign yields an SOF crossing at tested acceleration.")
    elif np.isnan(f_plus):
        sgn = -1.0
    elif np.isnan(f_minus):
        sgn = +1.0
    else:
        sgn = +1.0 if f_plus > f_minus else -1.0

    # exponentially expand the search bounds until a sign change is found in the b-plane difference function
    for _ in range(80):
        f = comp_b_diff(sgn * a_hi)
        if np.isnan(f):
            a_hi *= 2.0
            continue
        if f >= 0.0:
            break
        a_hi *= 2.0
    else:
        raise RuntimeError("Failed to find brackets for the required acceleration.")

    # use a root-finding method to solve for the required continuous acceleration that achieves the target b-plane miss distance
    a_min = sgn * float(brentq(lambda x: comp_b_diff(sgn * x), 0.0, a_hi, xtol=1e-12, rtol=1e-10, maxiter=200))

    # refine the solution to ensure it meets the target b-plane miss distance
    b_robust, cloud = cloud_at_piecewise(a_min)
    while b_robust < b_target:
        a_min *= 1.0000001
        b_robust, cloud = cloud_at_piecewise(a_min)
    worst_case = cloud["worst_case"]
    deltaV = a_min * max(worst_case["t_sof_rel"] - t_start, 0.0)

    return a_min, deltaV, b_robust, worst_case

# function to perform a coarse-to-fine search for the best continuous thrust start time to achieve a desired b-plane miss distance at SOF crossing with minimum required deltaV
def search_best_continuous_start_time_grss(sol_secondary, cov_secondary, mu_main, R_main, R_sof, direction, b_target, t_start_min, t_start_max, t_max=365.0, dt=1.0, burn_step=0.25, coarse_step=5.0, mid_half_width=10.0, mid_step=1.0, fine_half_width=2.0, fine_step=0.1, robust_metric="min_b", main_body_name="Earth", body_name="secondary"):
    best = None

    # define a function to try a grid of start times and update the best solution
    def try_time_grid(t_grid):
        nonlocal best
        for t_start in t_grid:
            try:
                a_min, deltaV, b_val, b_elems = solve4accel2perturb_continuous_at_time_grss(sol_secondary=sol_secondary, cov_secondary=cov_secondary, mu_main=mu_main, R_main=R_main, R_sof=R_sof, direction=direction, b_target=b_target, t_start=t_start, t_max=t_max, dt=dt, burn_step=burn_step, robust_metric=robust_metric, main_body_name=main_body_name, body_name=body_name)
            except Exception:
                continue
            if best is None or abs(deltaV) < abs(best["deltaV"]):
                best = {
                    "t_start": float(t_start),
                    "a_min": float(a_min),
                    "deltaV": float(deltaV),
                    "b": float(b_val),
                    "b_elems": b_elems,
                }

    # try the coarse grid of start times
    coarse_times = np.arange(t_start_min, t_start_max + 1e-12, coarse_step)
    try_time_grid(coarse_times)
    if best is None:
        raise RuntimeError("No feasible continuous-thrust start time found in coarse search.")

    # refine around the best coarse solution with a mid grid
    mid_min = max(t_start_min, best["t_start"] - mid_half_width)
    mid_max = min(t_start_max, best["t_start"] + mid_half_width)
    try_time_grid(np.arange(mid_min, mid_max + 1e-12, mid_step))

    # refine around the best mid solution with a fine grid
    fine_min = max(t_start_min, best["t_start"] - fine_half_width)
    fine_max = min(t_start_max, best["t_start"] + fine_half_width)
    try_time_grid(np.arange(fine_min, fine_max + 1e-12, fine_step))

    return best
