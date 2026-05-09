import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
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

# compute the local orbital basis (rhat, that, hhat) from an instantaneous heliocentric state
def get_rtn_basis(r, v):
    rhat = r / np.linalg.norm(r)
    hvec = np.cross(r, v)
    hhat = hvec / np.linalg.norm(hvec)
    that = np.cross(hhat, rhat)
    return rhat, that, hhat

# convert a GRSS solution dict for a secondary body to keplerian elements referenced to the current epoch
def grss_sol_to_kepler_elements(sol, mu):
    q = float(sol["q"])
    e = float(sol["e"])
    i = float(sol["i"])
    w = float(sol["w"])
    Omega = float(sol["om"])
    t = float(sol["t"])
    tp = float(sol["tp"])
    a = get_semimajor_axis(q, e)
    n = np.sqrt(mu / a**3)
    M = wrap_angle(n * (t - tp))
    return np.array([q, e, i, w, Omega, M], dtype=float)

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
def plot_bplane(xi, zeta, b_coll, R, title="Opik b-plane Plot", save=True, include_uncertainty=False, uncertainty_cloud=None, uncertainty_scale=1.0, zoom_uncertainty=False, zoom_pad=0.15):
    theta = np.linspace(0, 2*np.pi, 800)
    xi_n = xi / R
    zeta_n = zeta / R
    b_coll_n = b_coll / R
    fig, ax = plt.subplots(figsize=(7,7))
    ax.plot(np.cos(theta), np.sin(theta), label="Earth Radius")
    ax.plot(4*np.cos(theta), 4*np.sin(theta), label="4x Earth Radius")
    ax.plot(b_coll_n*np.cos(theta), b_coll_n*np.sin(theta), label="Collision Radius")
    max_extent = max(4, abs(xi_n), abs(zeta_n), abs(b_coll_n))
    xi_unc_n = None
    zeta_unc_n = None
    if include_uncertainty:
        if uncertainty_cloud is None:
            raise ValueError("include_uncertainty=True requires uncertainty_cloud.")
        clone_results = uncertainty_cloud.get("clone_results", [])
        if len(clone_results) > 0:
            xi_unc_n = np.array([uncertainty_scale * r["xi"] / R for r in clone_results], dtype=float)
            zeta_unc_n = np.array([uncertainty_scale * r["zeta"] / R for r in clone_results], dtype=float)
            xz_unc_n = np.column_stack((xi_unc_n, zeta_unc_n))
            mean_n = np.mean(xz_unc_n, axis=0)
            cov_n = np.cov(xz_unc_n, rowvar=False)
            add_covariance_ellipse(ax, mean=mean_n, cov=cov_n, nsig=3.0, label=r"$3\sigma$ B-plane covariance ellipse", zorder=2)
            ax.scatter(xi_unc_n, zeta_unc_n, s=28, alpha=0.65, color="red", label=r"$\sigma\text{-Point Encounters}$", zorder=3)
            max_extent = max(max_extent, float(np.max(np.abs(xi_unc_n))), float(np.max(np.abs(zeta_unc_n))))
    ax.scatter([xi_n], [zeta_n], marker="x", s=90, linewidths=2, color="blue", label="Encounter Point")
    pad = 1.1 * max_extent
    ax.set_xlim(-pad, pad)
    ax.set_ylim(-pad, pad)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(r"$\xi \; \text{(Earth Radii)}$")
    ax.set_ylabel(r"$\zeta \; \text{(Earth Radii)}$")
    ax.set_title(title)
    ax.grid(True)
    ax.legend(loc="best", frameon=True)
    if save:
        filename = f"{title}.png"
        fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    if zoom_uncertainty and include_uncertainty and xi_unc_n is not None and len(xi_unc_n) > 0:
        fig_zoom, ax_zoom = plt.subplots(figsize=(7,7))
        ax_zoom.plot(np.cos(theta), np.sin(theta), label="Earth Radius")
        ax_zoom.plot(4*np.cos(theta), 4*np.sin(theta), label="4x Earth Radius")
        ax_zoom.plot(b_coll_n*np.cos(theta), b_coll_n*np.sin(theta), label="Collision Radius")
        ax_zoom.scatter(xi_unc_n, zeta_unc_n, s=28, alpha=0.65, color="red", label=r"$\sigma\text{-Point Encounters}$")
        xz_unc_n_zoom = np.column_stack((xi_unc_n, zeta_unc_n))
        mean_n_zoom = np.mean(xz_unc_n_zoom, axis=0)
        cov_n_zoom = np.cov(xz_unc_n_zoom, rowvar=False)
        add_covariance_ellipse(ax_zoom, mean=mean_n_zoom, cov=cov_n_zoom, nsig=3.0, label=r"$3\sigma$ B-plane covariance ellipse", zorder=2)
        ax_zoom.scatter([xi_n], [zeta_n], marker="x", s=90, linewidths=2, color="blue", label="Encounter Point", zorder=3)
        x_all = np.append(xi_unc_n, xi_n)
        z_all = np.append(zeta_unc_n, zeta_n)
        x_min, x_max = np.min(x_all), np.max(x_all)
        z_min, z_max = np.min(z_all), np.max(z_all)
        x_center = 0.5 * (x_min + x_max)
        z_center = 0.5 * (z_min + z_max)
        x_span = x_max - x_min
        z_span = z_max - z_min
        long_span = max(x_span, z_span)
        min_span_fraction = 0.2
        x_span_eff = max(x_span, min_span_fraction * long_span)
        z_span_eff = max(z_span, min_span_fraction * long_span)
        x_half_range = 0.5 * x_span_eff + zoom_pad
        z_half_range = 0.5 * z_span_eff + zoom_pad
        ax_zoom.set_xlim(x_center - x_half_range, x_center + x_half_range)
        ax_zoom.set_ylim(z_center - z_half_range, z_center + z_half_range)
        ax_zoom.set_aspect('equal', adjustable='box')
        ax_zoom.set_xlabel(r"$\xi \; \text{(Earth Radii)}$")
        ax_zoom.set_ylabel(r"$\zeta \; \text{(Earth Radii)}$")
        ax_zoom.set_title(title)
        ax_zoom.grid(True)
        ax_zoom.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
        if save:
            filename_zoom = f"{title}_zoom.png"
            fig_zoom.savefig(filename_zoom, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig_zoom)

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
def compute_valsecchi_circles(rE, vE, rA, vA, mu1, mu2, R_planet, elements1, units_in_km=False, num_circs=81):

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
    
    res_list = []
    for p in range(2, num_circs):
        for q in range(1, p):
            if np.gcd(p, q) == 1:
                res_list.append((p, q))
    res_list = sorted(res_list, key=lambda pq: (pq[0] / pq[1], pq[0]))
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

# function to find the closest Valsecchi circles to a given b-plane point
def find_closest_valsecchi_circles(xi_point, zeta_point, rE, vE, rA, vA, mu1, mu2, R_planet, elements1, units_in_km=True, max_p=80, top_n=8, R_asteroid=0.0, ratio_min=1.0, ratio_max=2.0, target_b=None, target_b_tol=None,):

    # get the relative state
    r_rel = rA - rE
    v_rel = vA - vE

    # compute S_hat
    v_inf = np.linalg.norm(v_rel)
    S_hat = v_rel / v_inf

    # compute T_hat and R_hat
    T_hat = np.cross(vE, S_hat)
    T_hat = T_hat / np.linalg.norm(T_hat)
    R_hat = np.cross(S_hat, T_hat)

    # compute b-plane coordinates
    h = np.cross(r_rel, v_rel)
    B = np.cross(S_hat, h) / v_inf
    xi = float(np.dot(B, T_hat))
    zeta = float(np.dot(B, R_hat))

    # compute finite-size collision radius
    R_contact = R_planet + R_asteroid
    b_coll = R_contact * np.sqrt(1.0 + 2.0 * mu2 / (R_contact * v_inf**2))

    # convert to km if requested
    if units_in_km:
        xi = float(au2km(xi))
        zeta = float(au2km(zeta))
        b_coll = float(au2km(b_coll))
        v_inf_km_s = float(au_per_day2km_per_s(v_inf))
        rE_km = au2km(rE)
        vE_km_s = au_per_day2km_per_s(vE)
        mu1_km = float(au3_per_day2km3_per_s2(mu1))
        mu2_km = float(au3_per_day2km3_per_s2(mu2))
    else:
        v_inf_km_s = float(au_per_day2km_per_s(v_inf))
        rE_km = au2km(rE)
        vE_km_s = au_per_day2km_per_s(vE)
        mu1_km = float(au3_per_day2km3_per_s2(mu1))
        mu2_km = float(au3_per_day2km3_per_s2(mu2))

    # local heliocentric encounter frame
    r_hat = rE_km / np.linalg.norm(rE_km)
    v_hat_raw = vE_km_s / np.linalg.norm(vE_km_s)

    y_hat = v_hat_raw - np.dot(v_hat_raw, r_hat) * r_hat
    y_hat = y_hat / np.linalg.norm(y_hat)

    z_hat = np.cross(r_hat, y_hat)
    z_hat = z_hat / np.linalg.norm(z_hat)

    x_hat = r_hat

    # incoming asymptote components
    u_inf_minus = -v_inf_km_s * np.array(S_hat, dtype=float)

    Ux = float(np.dot(u_inf_minus, x_hat))
    Uy = float(np.dot(u_inf_minus, y_hat))
    Uz = float(np.dot(u_inf_minus, z_hat))
    U = float(np.linalg.norm(u_inf_minus))

    theta = float(np.arccos(np.clip(Uy / U, -1.0, 1.0)))
    phi = float(np.arctan2(Ux, Uz))

    c = float(mu2_km / (v_inf_km_s**2))

    # Earth/main-body reference orbit
    qE, eE, *_ = elements1
    aE = get_semimajor_axis(qE, eE)
    aE_km = float(au2km(aE))

    rE_mag_km = float(np.linalg.norm(rE_km))
    v_c = float(np.sqrt(mu1_km / rE_mag_km))
    U_ratio = float(v_inf_km_s / v_c)

    def theta_prime_for_resonance(p, q):
        a_res = aE_km * (p / q)**(2.0 / 3.0)
        v_res2 = mu1_km * (2.0 / rE_mag_km - 1.0 / a_res)
        cos_thp = (
            v_res2 / (v_c * v_c)
            - 1.0
            - U_ratio * U_ratio
        ) / (2.0 * U_ratio)
        return a_res, cos_thp

    def valsecchi_circle_params(theta, theta_p, c):
        denom = np.cos(theta_p) - np.cos(theta)
        D = c * np.sin(theta) / denom
        R = c * np.sin(theta_p) / denom
        return D, R

    def circle_to_target_ring_distance(D, Rc, target_b):
        d_centers = abs(D)
        r1 = abs(target_b)
        r2 = abs(Rc)
        if d_centers > r1 + r2:
            return d_centers - r1 - r2
        if d_centers < abs(r1 - r2):
            return abs(r1 - r2) - d_centers
        return 0.0

    # candidate resonance list
    candidates = []
    for p in range(2, max_p + 1):
        q_lo = max(1, int(np.floor(p / ratio_max)) - 2)
        q_hi = min(p - 1, int(np.ceil(p / ratio_min)) + 2)
        for q in range(q_lo, q_hi + 1):
            ratio = p / q
            if ratio < ratio_min or ratio > ratio_max:
                continue
            if np.gcd(p, q) != 1:
                continue
            a_res_km, cos_thp = theta_prime_for_resonance(p, q)
            if abs(cos_thp) > 1.0:
                continue
            thp = float(np.arccos(cos_thp))
            D_km, R_km = valsecchi_circle_params(theta, thp, c)
            if units_in_km:
                D = D_km
                Rc = R_km
                a_res = a_res_km
            else:
                D = float(km2au(D_km))
                Rc = float(km2au(R_km))
                a_res = float(km2au(a_res_km))

            # distance from point to this circle
            center_to_point = np.sqrt(xi_point**2 + (zeta_point - D)**2)
            dist_to_circle = abs(center_to_point - abs(Rc))
            signed_dist = center_to_point - abs(Rc)
            if target_b is not None:
                dist_to_target_ring = circle_to_target_ring_distance(D, Rc, target_b)
                if target_b_tol is not None and dist_to_target_ring > target_b_tol:
                    continue
            else:
                dist_to_target_ring = np.nan

            candidates.append({
                "p": p,
                "q": q,
                "ratio": p / q,
                "a_res": a_res,
                "theta_prime": thp,
                "D": D,
                "R": Rc,
                "distance_to_circle": float(dist_to_circle),
                "signed_distance_to_circle": float(signed_dist),
                "distance_to_target_ring": float(dist_to_target_ring),
                "circle_data": (p, q, a_res, thp, D, Rc),
            })

    if target_b is not None:
        candidates = sorted(candidates, key=lambda d: (d["distance_to_target_ring"], d["distance_to_circle"]))
    else:
        candidates = sorted(candidates, key=lambda d: d["distance_to_circle"])
    closest = candidates[:top_n]
    closest_circle_data = [c["circle_data"] for c in closest]

    return closest, closest_circle_data, xi, zeta, b_coll, theta, phi

# plot the b-plane visualization with Valsecchi circles
def plot_bplane_valsecchi(xi, zeta, b_coll, R, circle_data, title="Impact b-plane w/ Valsecchi Circles", save=True, units_in_km=False, include_uncertainty=False, uncertainty_cloud=None, uncertainty_scale=1.0):
    theta = np.linspace(0, 2*np.pi, 800)
    fig, ax = plt.subplots(figsize=(7,7))
    ax.plot(R*np.cos(theta), R*np.sin(theta), label="Earth Radius")
    ax.plot(4*R*np.cos(theta), 4*R*np.sin(theta), label="4x Earth Radius")
    ax.plot(b_coll*np.cos(theta), b_coll*np.sin(theta), label="Collision Radius")
    for k, (p, q, a_res, thp, D, Rc) in enumerate(circle_data):
        xi_c = Rc * np.sin(theta)
        zeta_c = D  + Rc * np.cos(theta)
        ax.plot(xi_c, zeta_c, linewidth=1.2, alpha=0.75, color="purple", label="Valsecchi Circles" if k == 0 else "_nolegend_")
    max_extent = max(4*R, abs(xi), abs(zeta), abs(b_coll))
    if include_uncertainty:
        if uncertainty_cloud is None:
            raise ValueError("include_uncertainty=True requires uncertainty_cloud.")
        clone_results = uncertainty_cloud.get("clone_results", [])
        if len(clone_results) > 0:
            if units_in_km:
                xi_unc = np.array([uncertainty_scale * au2km(r["xi"]) for r in clone_results], dtype=float)
                zeta_unc = np.array([uncertainty_scale * au2km(r["zeta"]) for r in clone_results], dtype=float)
            else:
                xi_unc = np.array([uncertainty_scale * r["xi"] for r in clone_results], dtype=float)
                zeta_unc = np.array([uncertainty_scale * r["zeta"] for r in clone_results], dtype=float)
            xz_unc = np.column_stack((xi_unc, zeta_unc))
            mean_unc = np.mean(xz_unc, axis=0)
            cov_unc = np.cov(xz_unc, rowvar=False)
            add_covariance_ellipse(ax, mean=mean_unc, cov=cov_unc, nsig=3.0, label=r"$3\sigma$ B-plane covariance ellipse", zorder=2)
            ax.scatter(xi_unc, zeta_unc, s=28, alpha=0.75, color="red", label=r"$\sigma\text{-Point Encounters}$", zorder=3)
            max_extent = max(max_extent, float(np.max(np.abs(xi_unc))), float(np.max(np.abs(zeta_unc))))
    ax.scatter([xi], [zeta], marker="x", s=80, label="Encounter Point")
    max_extent = max(max_extent, 4*R, abs(xi), abs(zeta))
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
    ax.legend(loc="best", frameon=True)
    if save:
        filename = f"{title}.png"
        fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

# extract the b-plane plot terms from a GRSS solution dict or a non-GRSS solution tuple
def extract_plot_terms(sol, use_GRSS_propagation):
    if use_GRSS_propagation:
        case = sol["b_elems"]
        xi = case["xi"]
        zeta = case["zeta"]
        b_coll = case["b_coll"]
    else:
        xi, zeta, _, b_coll = sol["b_elems"][:4]
    return xi, zeta, b_coll


### NON-GRSS FUNCTIONS ###

# solve for heliocentric state
def get_heliocentric_state(elements, mu, t, a_first=False):
    
    # extract the state
    if a_first:
        a, e, i, w, Omega, M = elements
    else:
        q, e, i, w, Omega, M = elements
        a = get_semimajor_axis(q, e)

    # compute mean motion
    n = np.sqrt(mu / a**3)

    # compute the mean anomaly at time t
    M = wrap_angle(M + n*t)

    # compute eccentric anomaly
    E = compute_E(M, e)

    # compute radial distance and true anomaly
    r_mag = a * (1 - e*np.cos(E))
    nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E/2), np.sqrt(1 - e) * np.cos(E/2))

    # compute perifocal position and velocity
    r_peri = np.array([r_mag*np.cos(nu), r_mag*np.sin(nu), 0.0])
    v_peri = np.sqrt(mu * a) / r_mag * np.array([-np.sin(E), np.sqrt(1 - e**2)*np.cos(E), 0.0])

    # rotate to inertial frame
    R_zxz = np.array([ 
        [np.cos(Omega)*np.cos(w) - np.sin(Omega)*np.cos(i)*np.sin(w) , -np.cos(Omega)*np.sin(w) - np.sin(Omega)*np.cos(i)*np.cos(w) , np.sin(Omega)*np.sin(i)], 
        [np.sin(Omega)*np.cos(w) + np.cos(Omega)*np.cos(i)*np.sin(w) , -np.sin(Omega)*np.sin(w) + np.cos(Omega)*np.cos(i)*np.cos(w), -np.cos(Omega)*np.sin(i)], 
        [np.sin(i)*np.sin(w) , np.sin(i)*np.cos(w), np.cos(i)] 
        ])
    r = R_zxz @ r_peri
    v = R_zxz @ v_peri

    return r, v

# compute gausses planetary equations
def compute_gauss_eqs(elements, mu, a_mag, direction):
    a, e, i, w, Omega, M = elements

    # choose acceleration direction (radial, transverse, normal)
    if direction == "radial":
        Fr, Ftheta, Fz = a_mag, 0.0, 0.0
    elif direction == "transverse":
        Fr, Ftheta, Fz = 0.0, a_mag, 0.0
    elif direction == "normal":
        Fr, Ftheta, Fz = 0.0, 0.0, a_mag
    else:
        raise ValueError("direction must be 'radial', 'transverse', or 'normal'")

    # compute eccentric anomaly and nu
    E = compute_E(M, e)
    nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E/2), np.sqrt(1 - e) * np.cos(E/2))

    # compute geometric terms
    p = a*(1 - e**2)
    r = p / (1 + e*np.cos(nu))
    h = np.sqrt(mu*p)
    n = np.sqrt(mu / a**3)

    # compute gauss planetary equations
    adot = a * (2*h/(mu*(1 - e**2))) * (e*np.sin(nu)*Fr + (1 + e*np.cos(nu))*Ftheta)
    Mdot = n + (h*np.sqrt(1 - e**2)/(mu*e)) * ((np.cos(nu) - (2*e/(1 - e**2))*(r/a))*Fr - (1 + (1/(1 - e**2))*(r/a))*np.sin(nu)*Ftheta)
    edot = (h/mu) * (np.sin(nu)*Fr + (np.cos(nu) + np.cos(E))*Ftheta)
    idot = (r*np.cos(w + nu)/h) * Fz
    wdot = -(h/(mu*e)) * (np.cos(nu)*Fr - ((2 + e*np.cos(nu))/(1 + e*np.cos(nu))) * np.sin(nu)*Ftheta) + -(np.cos(i) * r*np.sin(w + nu)/(h*np.sin(i))) * Fz
    Omegadot = (r*np.sin(w + nu)/(h*np.sin(i))) * Fz

    return np.array([adot, edot, idot, wdot, Omegadot, Mdot])

# propagate the orbital elements under a small acceleration using Gauss' planetary equations
def propagate_gauss(elements, mu, t, a_mag, direction, a_first=False):

    # extract the state
    if a_first:
        a, e, i, w, Omega, M = elements
    else:
        q, e, i, w, Omega, M = elements
        a = get_semimajor_axis(q, e)

    # define the ODE function for the planetary equations
    def gauss_odes(t, y): return compute_gauss_eqs(y, mu, a_mag, direction)

    # solve the ODEs
    sol = solve_ivp(gauss_odes, [0.0, t], [a, e, i, w, Omega, M], method='RK45', rtol=1e-6)

    # extract the final state
    a_final = sol.y[0][-1]
    e_final = sol.y[1][-1]
    i_final = wrap_angle(sol.y[2][-1])
    w_final = wrap_angle(sol.y[3][-1])
    Omega_final = wrap_angle(sol.y[4][-1])
    M_final = wrap_angle(sol.y[5][-1])

    # return the final state in the same format as the input
    if a_first:
        return np.array([a_final, e_final, i_final, w_final, Omega_final, M_final])
    else:
        q_final = a_final * (1 - e_final)
        return np.array([q_final, e_final, i_final, w_final, Omega_final, M_final])
    
# define a function to get the heliocentric state without advancing the mean anomaly
def get_heliocentric_state_now(elements, mu, a_first=False):
    
    # extract the state
    if a_first:
        a, e, i, w, Omega, M = elements
    else:
        q, e, i, w, Omega, M = elements
        a = get_semimajor_axis(q, e)

    # compute eccentric anomaly
    E = compute_E(M, e)

    # compute radial distance and true anomaly
    r_mag = a * (1 - e*np.cos(E))
    nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E/2), np.sqrt(1 - e) * np.cos(E/2))

    # compute perifocal position and velocity
    r_peri = np.array([r_mag*np.cos(nu), r_mag*np.sin(nu), 0.0])
    v_peri = np.sqrt(mu * a) / r_mag * np.array([-np.sin(E), np.sqrt(1 - e**2)*np.cos(E), 0.0])

    # rotate to inertial frame
    R_zxz = np.array([ 
        [np.cos(Omega)*np.cos(w) - np.sin(Omega)*np.cos(i)*np.sin(w) , -np.cos(Omega)*np.sin(w) - np.sin(Omega)*np.cos(i)*np.cos(w) , np.sin(Omega)*np.sin(i)], 
        [np.sin(Omega)*np.cos(w) + np.cos(Omega)*np.cos(i)*np.sin(w) , -np.sin(Omega)*np.sin(w) + np.cos(Omega)*np.cos(i)*np.cos(w), -np.cos(Omega)*np.sin(i)], 
        [np.sin(i)*np.sin(w) , np.sin(i)*np.cos(w), np.cos(i)] 
        ])
    r = R_zxz @ r_peri
    v = R_zxz @ v_peri

    return r, v

# function to convert heliocentric state to orbital elements referenced to the current epoch
def helio2elems(r, v, mu):

    # compute standard elems
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    rmag = np.linalg.norm(r)
    vmag = np.linalg.norm(v)
    h = np.cross(r, v)
    hmag = np.linalg.norm(h)
    i = np.arccos(np.clip(h[2] / hmag, -1.0, 1.0))

    # node line
    k = np.array([0.0, 0.0, 1.0])
    nvec = np.cross(k, h)
    nmag = np.linalg.norm(nvec)

    # eccentricity vector
    evec = (1.0 / mu) * (np.cross(v, h) - mu * (r / rmag))
    e = float(np.linalg.norm(evec))
    if not (e < 1.0): raise ValueError(f"implementation requires elliptic orbit (e<1). Got e={e}.")

    # semi-major axis
    a = -mu / (2.0 * (0.5 * vmag**2 - mu / rmag))

    # RAAN
    if nmag > 1e-15: Omega = np.arctan2(nvec[1], nvec[0])
    else: Omega = 0.0

    # argument of periapsis
    if nmag > 1e-15 and e > 1e-15:
        w = np.arctan2(np.dot(np.cross(nvec, evec), h) / (nmag * e * hmag), np.dot(nvec, evec) / (nmag * e))
    else:
        w = 0.0

    # true anomaly (nu)
    if e > 1e-15:
        nu = np.arctan2(np.dot(np.cross(evec, r), h) / (e * hmag * rmag), np.dot(evec, r) / (e * rmag))
    else:
        if nmag > 1e-15:
            nu = np.arctan2(np.dot(np.cross(nvec, r), h) / (nmag * hmag * rmag), np.dot(nvec, r) / (nmag * rmag))
        else:
            nu = np.arctan2(r[1], r[0])

    # eccentric anomaly E from nu (elliptic)
    E = 2.0 * np.arctan2(np.sqrt(1.0 - e) * np.sin(nu / 2.0), np.sqrt(1.0 + e) * np.cos(nu / 2.0))
    E = wrap_angle(E)

    # mean anomaly
    M = wrap_angle(E - e * np.sin(E))

    # periapsis distance q
    q = a * (1.0 - e)

    return np.array([q, e, wrap_angle(i), wrap_angle(w), wrap_angle(Omega), M], dtype=float)

# function to convert an orbit from one t_epoch to a new t_epoch
def re_epoch_elements(elements, mu, t_epoch):
    r_epoch, v_epoch = get_heliocentric_state(elements, mu, t_epoch, a_first=False)
    return helio2elems(r_epoch, v_epoch, mu)

# function to apply an impulse at an arbitrary absolute time t_impulse
def apply_impulse_to_elements_at_time(elements, mu, t_impulse, dV_signed, direction):

    # get state at impulse time and define directions
    r_imp, v_imp = get_heliocentric_state(elements, mu, t_impulse, a_first=False)
    rhat = r_imp / np.linalg.norm(r_imp)
    hvec = np.cross(r_imp, v_imp)
    hhat = hvec / np.linalg.norm(hvec)
    that = np.cross(hhat, rhat)

    if direction == "radial":
        dV_vec = dV_signed * rhat
    elif direction == "transverse":
        dV_vec = dV_signed * that
    elif direction == "normal":
        dV_vec = dV_signed * hhat
    else:
        raise ValueError("direction must be 'radial', 'transverse', or 'normal'")

    # apply impulse in direction and re-epoch to t_impulse
    v_perturbed = v_imp + dV_vec
    return helio2elems(r_imp, v_perturbed, mu)

# solve for the required instantaneous deltaV, with the impulse applied at absolute time t_impulse
def solve4accel2perturb_instantaneous_at_time_nongrss(elements_main, elements_secondary, mu_sun, mu_main, R_main, direction, b_target, t_impulse, t_eval, t_max=365.0):

    # define a function to compute the b-plane at the fixed evaluation time
    def b_at_fixed_eval_impulse(dV_signed):
        elements2_post = apply_impulse_to_elements_at_time(elements_secondary, mu_sun, t_impulse, dV_signed, direction)
        if t_eval < t_impulse:
            return None, None
        r1, v1 = get_heliocentric_state(elements_main, mu_sun, t_eval, a_first=False)
        r2, v2 = get_heliocentric_state(elements2_post, mu_sun, t_eval - t_impulse, a_first=False)
        xi, zeta, B, b_coll, S_hat, T_hat, R_hat, v_inf = get_bplane_now(r1, v1, r2, v2, mu_main, R_main)
        b = float(np.linalg.norm(B))
        return b, (xi, zeta, B, b_coll, S_hat, T_hat, R_hat, v_inf, elements2_post, t_eval, t_impulse)

    # define a function to compute the difference between the current b-plane and the target b-plane
    def comp_b_diff(dV_signed):
        b, _ = b_at_fixed_eval_impulse(dV_signed)
        if b is None:
            return np.nan
        return b - b_target

    # ensure asteroid does not already exceed target b-plane
    f0 = comp_b_diff(0.0)
    if not np.isnan(f0) and f0 >= 0.0:
        b0, b_elems0 = b_at_fixed_eval_impulse(0.0)
        return 0.0, b0, b_elems0

    # pick the acceleration sign that closes the target b-plane distance
    dV_hi = 1e-12
    f_plus = comp_b_diff(+dV_hi)
    f_minus = comp_b_diff(-dV_hi)
    if np.isnan(f_plus) and np.isnan(f_minus):
        raise RuntimeError("Neither sign reaches the target b-plane distance at the GRSS evaluation time.")
    elif np.isnan(f_plus):
        sgn = -1.0
    elif np.isnan(f_minus):
        sgn = +1.0
    else:
        sgn = +1.0 if f_plus > f_minus else -1.0

    # solve for deltaV via bracketing
    for _ in range(80):
        f = comp_b_diff(sgn * dV_hi)
        if np.isnan(f):
            dV_hi *= 2.0
            continue
        if f >= 0.0:
            break
        dV_hi *= 2.0
    else:
        raise RuntimeError("Failed to bracket the required impulse deltaV.")

    # solve for the minimum required deltaV
    dV_min = sgn * float(brentq(lambda x: comp_b_diff(sgn * x), 0.0, dV_hi, xtol=1e-12, rtol=1e-10, maxiter=200))

    # check for numerical feasibility
    b, b_elems = b_at_fixed_eval_impulse(dV_min)
    while b < b_target:
        dV_min *= 1.0000001
        b, b_elems = b_at_fixed_eval_impulse(dV_min)

    return dV_min, b, b_elems

# solve for the required continuous acceleration with thrust starting at time t_start
def solve4accel2perturb_continuous_at_time_nongrss(elements_main, elements_secondary, mu_sun, mu_main, R_main, direction, b_target, t_start, t_eval, t_max=365.0):
    elements2_start = re_epoch_elements(elements_secondary, mu_sun, t_start)

    # define a function to apply a constant acceleration starting at t_start and compute the b-plane at the fixed evaluation time
    def b_at_fixed_eval_constant_accel(a_mag):
        if t_eval <= t_start:
            elems2_eval = elements2_start
            r2, v2 = get_heliocentric_state_now(elems2_eval, mu_sun, a_first=False)
        else:
            elems2_eval = propagate_gauss(elements2_start, mu_sun, t_eval - t_start, a_mag, direction, a_first=False)
            r2, v2 = get_heliocentric_state_now(elems2_eval, mu_sun, a_first=False)
        r1, v1 = get_heliocentric_state(elements_main, mu_sun, t_eval, a_first=False)
        xi, zeta, B, b_coll, S_hat, T_hat, R_hat, v_inf = get_bplane_now(r1, v1, r2, v2, mu_main, R_main)
        b = float(np.linalg.norm(B))
        return b, (xi, zeta, B, b_coll, S_hat, T_hat, R_hat, v_inf, elems2_eval, t_eval, t_start)

    # define a function to compute the difference between the current b-plane and the target b-plane
    def comp_b_diff(a_signed):
        b, _ = b_at_fixed_eval_constant_accel(a_signed)
        return b - b_target

    # ensure asteroid does not already exceed target b-plane
    f0 = comp_b_diff(0.0)
    if not np.isnan(f0) and f0 >= 0.0:
        b0, b_elems0 = b_at_fixed_eval_constant_accel(0.0)
        return 0.0, 0.0, b0, b_elems0

    # pick the acceleration sign that closes the target b-plane distance
    a_hi = 1e-14
    f_plus = comp_b_diff(+a_hi)
    f_minus = comp_b_diff(-a_hi)
    if np.isnan(f_plus) and np.isnan(f_minus):
        raise RuntimeError("Neither sign reaches the target b-plane distance at the GRSS evaluation time.")
    elif np.isnan(f_plus):
        sgn = -1.0
    elif np.isnan(f_minus):
        sgn = +1.0
    else:
        sgn = +1.0 if f_plus > f_minus else -1.0

    # solve for deltaV via bracketing
    for _ in range(80):
        f = comp_b_diff(sgn * a_hi)
        if np.isnan(f):
            a_hi *= 2.0
            continue
        if f >= 0.0:
            break
        a_hi *= 2.0
    else:
        raise RuntimeError("Failed to bracket the required continuous acceleration.")

    # solve for the minimum required deltaV
    a_min = sgn * float(brentq(lambda a: comp_b_diff(sgn * a), 0.0, a_hi, xtol=1e-12, rtol=1e-10, maxiter=200))

    # check for numerical feasibility
    b, b_elems = b_at_fixed_eval_constant_accel(a_min)
    while b < b_target:
        a_min *= 1.0000001
        b, b_elems = b_at_fixed_eval_constant_accel(a_min)

    deltaV = a_min * max(t_eval - t_start, 0.0)

    return a_min, deltaV, b, b_elems

# coarse-to-fine search for the best instantaneous-impulse time
def search_best_impulse_time_nongrss(elements_main, elements_secondary, mu_sun, mu_main, R_main, direction, b_target, t_impulse_min, t_impulse_max, t_eval, t_max=365.0, coarse_step=5.0, mid_half_width=10.0, mid_step=1.0, fine_half_width=2.0, fine_step=0.1):
    best = None

    # define a function to try a grid of impulse times and update the best solution
    def try_time_grid(t_grid):
        nonlocal best
        for t_impulse in t_grid:
            try:
                dV_min, b_val, b_elems = solve4accel2perturb_instantaneous_at_time_nongrss(elements_main=elements_main, elements_secondary=elements_secondary, mu_sun=mu_sun, mu_main=mu_main, R_main=R_main, direction=direction, b_target=b_target, t_impulse=t_impulse, t_eval=t_eval, t_max=t_max)
            except Exception:
                continue
            if best is None or abs(dV_min) < abs(best["dV_min"]):
                best = {"t_impulse": float(t_impulse), "dV_min": float(dV_min), "b": float(b_val), "b_elems": b_elems}

    # try the coarse grid of impulse times
    coarse_times = np.arange(t_impulse_min, t_impulse_max + 1e-12, coarse_step)
    try_time_grid(coarse_times)
    if best is None: raise RuntimeError("No feasible impulse time found in coarse search.")

    # refine around the best coarse solution with a mid grid
    mid_min = max(t_impulse_min, best["t_impulse"] - mid_half_width)
    mid_max = min(t_impulse_max, best["t_impulse"] + mid_half_width)
    try_time_grid(np.arange(mid_min, mid_max + 1e-12, mid_step))

    # refine around the best mid solution with a fine grid
    fine_min = max(t_impulse_min, best["t_impulse"] - fine_half_width)
    fine_max = min(t_impulse_max, best["t_impulse"] + fine_half_width)
    try_time_grid(np.arange(fine_min, fine_max + 1e-12, fine_step))

    return best

# coarse-to-fine search for the best continuous-thrust start time
def search_best_continuous_start_time_nongrss(elements_main, elements_secondary, mu_sun, mu_main, R_main, direction, b_target, t_start_min, t_start_max, t_eval, t_max=365.0, coarse_step=5.0, mid_half_width=10.0, mid_step=1.0, fine_half_width=2.0, fine_step=0.1):
    best = None

    # define a function to try a grid of start times and update the best solution
    def try_time_grid(t_grid):
        nonlocal best
        for t_start in t_grid:
            try:
                a_min, deltaV, b_val, b_elems = solve4accel2perturb_continuous_at_time_nongrss(elements_main=elements_main, elements_secondary=elements_secondary, mu_sun=mu_sun, mu_main=mu_main, R_main=R_main, direction=direction, b_target=b_target, t_start=t_start, t_eval=t_eval, t_max=t_max)
            except Exception:
                continue
            if best is None or abs(deltaV) < abs(best["deltaV"]):
                best = {"t_start": float(t_start), "a_min": float(a_min), "deltaV": float(deltaV), "b": float(b_val), "b_elems": b_elems}

    # try the coarse grid of start times
    coarse_times = np.arange(t_start_min, t_start_max + 1e-12, coarse_step)
    try_time_grid(coarse_times)
    if best is None: raise RuntimeError("No feasible continuous-thrust start time found in coarse search.")

    # refine around the best coarse solution with a mid grid
    mid_min = max(t_start_min, best["t_start"] - mid_half_width)
    mid_max = min(t_start_max, best["t_start"] + mid_half_width)
    try_time_grid(np.arange(mid_min, mid_max + 1e-12, mid_step))

    # refine around the best mid solution with a fine grid
    fine_min = max(t_start_min, best["t_start"] - fine_half_width)
    fine_max = min(t_start_max, best["t_start"] + fine_half_width)
    try_time_grid(np.arange(fine_min, fine_max + 1e-12, fine_step))

    return best


### GRSS-SPECIFIC FUNCTIONS ###

# function to initialize a GRSS sim from a GRSS solution dict for a secondary body
def init_grss_sim(sol_secondary, tf_abs, body_name="secondary"):
    dekernel_dir = str(utils.default_kernel_path)
    if not os.path.isdir(dekernel_dir):
        raise FileNotFoundError(f"GRSS kernel directory not found: {dekernel_dir}")
    expected = os.path.join(dekernel_dir, "de440.bsp")
    if not os.path.isfile(expected):
        raise FileNotFoundError(f"Expected DE440 kernel not found: {expected}")
    asteroid = prop.IntegBody(name=body_name, t0=float(sol_secondary["t"]), mass=0.0, radius=0.0, cometaryState=grss_cometary_state(sol_secondary), ngParams=prop.NongravParameters())
    sim = prop.PropSimulation(name=body_name, t0=float(sol_secondary["t"]), defaultSpiceBodies=440, DEkernelPath=dekernel_dir)
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

    # nominal-only path (no covariance cloud, one sim only)
    if cov_secondary is None:
        sim = init_grss_sim(sol_secondary, tf_abs, body_name=body_name)
        for evt in events:
            sim.add_event(evt)
        sim.integrate()
        sim.map_ephemeris()
        t_sof_abs = find_sof_crossing_time_grss(sim=sim, R_sof=R_sof, t_search_start_abs=t_search_start_abs, tf_abs=tf_abs, dt_days=dt, main_body_name=main_body_name)
        if t_sof_abs is None:
            return None
        r_main, v_main, r_secondary, v_secondary = get_states_grss(sim, t_sof_abs, main_body_name=main_body_name)
        xi, zeta, B, b_coll, S_hat, T_hat, R_hat, v_inf = get_bplane_now(r_main, v_main, r_secondary, v_secondary, mu_main, R_main)
        result = {
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
        }
        return {
            "sigma_points": None,
            "clone_results": [result],
            "b_vals": np.array([result["b"]], dtype=float),
            "b_robust": float(result["b"]),
            "worst_case": result,
        }

    # covariance-cloud path
    sigma_points = prop.SigmaPoints(sol_secondary, cov_secondary, "merwe", 1e-3, 2.0, 0.0)
    clones = sigma_points.sigma_points_dict
    if hasattr(clones, "values"): clone_iter = clones.values()
    else: clone_iter = clones
    clone_results = []
    for clone in clone_iter:
        clone_sol = clone_sol_dict(clone)
        if clone_sol["t"] is None:
            clone_sol["t"] = t0_abs
        sim = init_grss_sim(clone_sol, tf_abs, body_name=body_name)
        for evt in events:
            sim.add_event(evt)
        sim.integrate()
        sim.map_ephemeris()
        t_sof_abs = find_sof_crossing_time_grss(sim=sim, R_sof=R_sof, t_search_start_abs=t_search_start_abs, tf_abs=tf_abs, dt_days=dt, main_body_name=main_body_name)
        if t_sof_abs is None:
            clone_results.append(None)
            continue
        r_main, v_main, r_secondary, v_secondary = get_states_grss(sim, t_sof_abs, main_body_name=main_body_name)
        xi, zeta, B, b_coll, S_hat, T_hat, R_hat, v_inf = get_bplane_now(r_main, v_main, r_secondary, v_secondary, mu_main, R_main)
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
    feasible = [r for r in clone_results if r is not None]
    if not feasible:
        return None
    b_vals = np.array([r["b"] for r in feasible], dtype=float)
    xz_vals = np.array([[r["xi"], r["zeta"]] for r in feasible], dtype=float)
    bplane_mean = np.mean(xz_vals, axis=0)
    bplane_cov = np.cov(xz_vals, rowvar=False)
    if robust_metric == "min_b":
        b_robust = float(np.min(b_vals))
    elif robust_metric == "mean_b":
        b_robust = float(np.mean(b_vals))
    elif robust_metric == "p10_b":
        b_robust = float(np.percentile(b_vals, 10.0))
    else:
        raise ValueError("robust_metric must be 'min_b', 'mean_b', or 'p10_b'")
    worst_idx = int(np.argmin(b_vals))
    worst_case = feasible[worst_idx]
    return {
        "sigma_points": sigma_points,
        "clone_results": feasible,
        "b_vals": b_vals,
        "b_robust": b_robust,
        "worst_case": worst_case,
        "bplane_mean": bplane_mean,
        "bplane_cov": bplane_cov,
    }

# function to add a covariance ellipse to a matplotlib axis given a mean and covariance in the b-plane
def add_covariance_ellipse(ax, mean, cov, nsig=3.0, label=None, edgecolor="red", facecolor="red", alpha=0.25, linewidth=1.5, zorder=2):
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vals = np.maximum(vals, 0.0)
    vecs = vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width = 2.0 * nsig * np.sqrt(vals[0])
    height = 2.0 * nsig * np.sqrt(vals[1])
    ell = Ellipse(xy=mean, width=width, height=height, angle=angle, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, linewidth=linewidth, label=label if label is not None else f"{nsig:.0f}$\\sigma$ B-plane covariance ellipse")
    ax.add_patch(ell)
    return ell

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
        return 0.0, b0, cloud0["worst_case"], cloud0

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

    return dV_min, b_robust, cloud["worst_case"], cloud

# function to perform a coarse-to-fine search for the best impulse time to achieve a desired b-plane miss distance at SOF crossing with minimum required deltaV
def search_best_impulse_time_grss(sol_secondary, cov_secondary, mu_main, R_main, R_sof, direction, b_target, t_impulse_min, t_impulse_max, t_max=365.0, dt=1.0, coarse_step=5.0, mid_half_width=10.0, mid_step=1.0, fine_half_width=2.0, fine_step=0.1, robust_metric="min_b", main_body_name="Earth", body_name="secondary"):
    best = None

    # define a function to try a grid of impulse times and update the best solution
    def try_time_grid(t_grid):
        nonlocal best
        for t_impulse in t_grid:
            try:
                dV_min, b_val, b_elems, _ = solve4accel2perturb_instantaneous_at_time_grss(sol_secondary=sol_secondary, cov_secondary=cov_secondary, mu_main=mu_main, R_main=R_main, R_sof=R_sof, direction=direction, b_target=b_target, t_impulse=t_impulse, t_max=t_max, dt=dt, robust_metric=robust_metric, main_body_name=main_body_name, body_name=body_name)
            except Exception as e:
                print(f"t_impulse = {t_impulse:.6f} failed: {e}")
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
            except Exception as e:
                print(f"t_start = {t_start:.6f} failed: {e}")
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
