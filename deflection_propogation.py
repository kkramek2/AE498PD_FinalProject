import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton, brentq
from scipy.integrate import solve_ivp

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

# compute two-body distance
def compute_body_dist(elements1, elements2, mu, t, buffer):
    r1, _ = get_heliocentric_state(elements1, mu, t)
    r2, _ = get_heliocentric_state(elements2, mu, t)
    dist = float(np.linalg.norm(r2 - r1) - buffer)
    return dist

# compute time to impact
def determine_impact(elements1, elements2, mu, buffer, t_max, dt):

    # define callable for time refinement function
    def d(t): return compute_body_dist(elements1, elements2, mu, t, buffer)

    # step forward through time to find collision
    t_impact = None
    t = 0.0
    t_prev = t
    d_prev = d(t_prev)
    while t <= t_max:
        t += dt

        # compute the distance between both bodies
        d_curr = d(t)

        # return the time of impact
        if (d_prev > 0.0) and (d_curr <= 0.0): 
            t_impact = float(brentq(d, t_prev, t))
            return t_impact
        else:
            t_prev = t
            d_prev = d_curr
    
    # return nothing if no impact is found
    return t_impact

# compute the relative state (planetocentric)
def get_relative_state(elements1, elements2, mu, t):
    r1, v1 = get_heliocentric_state(elements1, mu, t)
    r2, v2 = get_heliocentric_state(elements2, mu, t)
    r = r2 - r1
    v = v2 - v1
    return r, v

# compute the Opik b-plane coordinates
def get_bplane_state(elements1, elements2, mu1, mu2, R_planet, t):

    # get the relative state
    r_rel, v_rel = get_relative_state(elements1, elements2, mu1, t)

    # compute S_hat
    v_inf = np.linalg.norm(v_rel)
    S_hat = v_rel / v_inf

    # compute T_hat
    _, v1 = get_heliocentric_state(elements1, mu1, t)
    T_hat = np.cross(v1, S_hat) / np.linalg.norm(np.cross(v1, S_hat))

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

    return xi, zeta, B, b_coll, S_hat, T_hat, R_hat, v_inf

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

# function to find the first time the relative distance crosses R_sof, starting the search at t_search_start
def find_sof_crossing_time(state2_func, elements1, mu1, R_sof, t_search_start, t_max, dt):
    def dist2sof(t):
        r1, _ = get_heliocentric_state(elements1, mu1, t)
        r2, _ = state2_func(t)
        return float(np.linalg.norm(r2 - r1) - R_sof)
    t_prev = max(0.0, t_search_start)
    d_prev = dist2sof(t_prev)
    if d_prev <= 0.0:
        return t_prev
    t = t_prev
    while t <= t_max:
        t += dt
        d_curr = dist2sof(t)
        if (d_prev > 0.0) and (d_curr <= 0.0):
            return float(brentq(dist2sof, t_prev, t))
        t_prev = t
        d_prev = d_curr
    return None

# solve for the required continuous acceleration with thrust starting at time t_start
def solve4accel2perturb_continuous_at_time(elements1, elements2, mu1, mu2, R_planet, R_sof, direction, b_target, t_start, t_max=365.0, dt=1.0):
    elements2_start = re_epoch_elements(elements2, mu1, t_start)

    # define a function to apply a constant acceleration starting at t_start and compute the b-plane at SOF crossing
    def b_at_sof_constant_accel(a_mag):

        def state2_func(t):
            if t <= t_start:
                return get_heliocentric_state(elements2, mu1, t, a_first=False)

            elements2_t = propagate_gauss(elements2_start, mu1, t - t_start, a_mag, direction, a_first=False)
            return get_heliocentric_state_now(elements2_t, mu1, a_first=False)

        t_sof = find_sof_crossing_time(state2_func, elements1, mu1, R_sof, t_start, t_max, dt)
        if t_sof is None:
            return None, None, None

        r1, v1 = get_heliocentric_state(elements1, mu1, t_sof)
        if t_sof <= t_start:
            elems2_sof = elements2_start
            r2, v2 = get_heliocentric_state_now(elems2_sof, mu1, a_first=False)
        else:
            elems2_sof = propagate_gauss(elements2_start, mu1, t_sof - t_start, a_mag, direction, a_first=False)
            r2, v2 = get_heliocentric_state_now(elems2_sof, mu1, a_first=False)

        xi, zeta, B, b_coll, S_hat, T_hat, R_hat, v_inf = get_bplane_now(r1, v1, r2, v2, mu2, R_planet)
        b = float(np.linalg.norm(B))
        return b, t_sof, (xi, zeta, B, b_coll, S_hat, T_hat, R_hat, v_inf, elems2_sof, t_sof, t_start)

    # define a function to compute the difference between the current b-plane and the target b-plane
    def comp_b_diff(a_signed):
        b, _, _ = b_at_sof_constant_accel(a_signed)
        if b is None:
            return np.nan
        return b - b_target

    # ensure asteroid does not already exceed target b-plane
    f0 = comp_b_diff(0.0)
    if not np.isnan(f0) and f0 >= 0.0:
        b0, _, b_elems0 = b_at_sof_constant_accel(0.0)
        return 0.0, 0.0, b0, b_elems0

    # pick the acceleration sign that closes the target b-plane distance
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
        raise RuntimeError("Failed to find brackets for the required acceleration.")

    # solve for the minimum required deltaV
    a_lo = 0.0
    a_min = sgn * float(brentq(lambda a: comp_b_diff(sgn * a), a_lo, a_hi, xtol=1e-12, rtol=1e-10, maxiter=200))

    # check for numerical feasibility (ensure dV_min does not yield a b_min below b_target)
    b, t_sof, b_elems = b_at_sof_constant_accel(a_min)
    while b < b_target:
        a_min *= 1.0000001
        b, t_sof, b_elems = b_at_sof_constant_accel(a_min)

    deltaV = a_min * max(t_sof - t_start, 0.0)

    return a_min, deltaV, b, b_elems

# solve for the required instantaneous deltaV, with the impulse applied at absolute time t_impulse
def solve4accel2perturb_instantaneous_at_time(elements1, elements2, mu1, mu2, R_planet, R_sof, direction, b_target, t_impulse, t_max=365.0, dt=1.0):

    # define a function to compute the b-plane at t_sof
    def b_at_sof_impulse(dV_signed):
        elements2_post = apply_impulse_to_elements_at_time(elements2, mu1, t_impulse, dV_signed, direction)

        def state2_func(t):
            if t < t_impulse:
                return get_heliocentric_state(elements2, mu1, t, a_first=False)
            return get_heliocentric_state(elements2_post, mu1, t - t_impulse, a_first=False)

        t_sof = find_sof_crossing_time(state2_func, elements1, mu1, R_sof, t_impulse, t_max, dt)
        if t_sof is None:
            return None, None, None

        r1, v1 = get_heliocentric_state(elements1, mu1, t_sof)
        r2, v2 = get_heliocentric_state(elements2_post, mu1, max(t_sof - t_impulse, 0.0), a_first=False)
        xi, zeta, B, b_coll, S_hat, T_hat, R_hat, v_inf = get_bplane_now(r1, v1, r2, v2, mu2, R_planet)
        b = float(np.linalg.norm(B))

        return b, t_sof, (xi, zeta, B, b_coll, S_hat, T_hat, R_hat, v_inf, elements2_post, t_sof, t_impulse)

    # define a function to compute the difference between the current b-plane and the target b-plane
    def comp_b_diff(dV_signed):
        b, _, _ = b_at_sof_impulse(dV_signed)
        if b is None:
            return np.nan
        return b - b_target

    # ensure asteroid does not already exceed target b-plane
    f0 = comp_b_diff(0.0)
    if not np.isnan(f0) and f0 >= 0.0:
        b0, _, b_elems0 = b_at_sof_impulse(0.0)
        return 0.0, b0, b_elems0

    # pick the acceleration sign that closes the target b-plane distance
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
        raise RuntimeError("Failed to find brackets for the required deltaV.")

    # solve for the minimum required deltaV
    dV_lo = 0.0
    dV_min = sgn * float(brentq(lambda x: comp_b_diff(sgn * x), dV_lo, dV_hi, xtol=1e-12, rtol=1e-10, maxiter=200))

    # check for numerical feasibility (ensure dV_min does not yield a b_min below b_target)
    b, _, b_elems = b_at_sof_impulse(dV_min)
    while b < b_target:
        dV_min *= 1.0000001
        b, _, b_elems = b_at_sof_impulse(dV_min)

    return dV_min, b, b_elems

# coarse-to-fine search for the best continuous-thrust start time
def search_best_continuous_start_time(elements1, elements2, mu1, mu2, R_planet, R_sof, direction, b_target, t_start_min, t_start_max, t_max=365.0, dt=1.0, coarse_step=5.0, mid_half_width=10.0, mid_step=1.0, fine_half_width=2.0, fine_step=0.1):
    best = None

    # define a function to try a grid of start times and update the best solution
    def try_time_grid(t_grid):
        nonlocal best
        for t_start in t_grid:
            try:
                a_min, deltaV, b_val, b_elems = solve4accel2perturb_continuous_at_time(elements1=elements1, elements2=elements2, mu1=mu1, mu2=mu2, R_planet=R_planet, R_sof=R_sof, direction=direction, b_target=b_target, t_start=t_start, t_max=t_max, dt=dt)
            except Exception:
                continue

            # update best solution if this start time yields a smaller required deltaV than the current best
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
    mid_times = np.arange(mid_min, mid_max + 1e-12, mid_step)
    try_time_grid(mid_times)

    # refine around the best mid solution with a fine grid
    fine_min = max(t_start_min, best["t_start"] - fine_half_width)
    fine_max = min(t_start_max, best["t_start"] + fine_half_width)
    fine_times = np.arange(fine_min, fine_max + 1e-12, fine_step)
    try_time_grid(fine_times)

    return best

# coarse-to-fine search for the best instantaneous-impulse time
def search_best_impulse_time(elements1, elements2, mu1, mu2, R_planet, R_sof, direction, b_target, t_impulse_min, t_impulse_max, t_max=365.0, dt=1.0, coarse_step=5.0, mid_half_width=10.0, mid_step=1.0, fine_half_width=2.0, fine_step=0.1):
    best = None

    # define a function to try a grid of impulse times and update the best solution
    def try_time_grid(t_grid):
        nonlocal best
        for t_impulse in t_grid:
            try:
                dV_min, b_val, b_elems = solve4accel2perturb_instantaneous_at_time(elements1=elements1, elements2=elements2, mu1=mu1, mu2=mu2, R_planet=R_planet, R_sof=R_sof, direction=direction, b_target=b_target, t_impulse=t_impulse, t_max=t_max, dt=dt)
            except Exception:
                continue

            # update best solution if this impulse time yields a smaller required deltaV than the current best
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
    