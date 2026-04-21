import numpy as np
import spiceypy as spice
import matplotlib.pyplot as plt
spice.furnsh('/Users/jadendowning/spice_kernels/meta.tm')

#Initial orbital elements given in the problem statement (t=0)
mu = 1.3271244e11       #km^3/s^2, gravitational parameter for the Sun
mu_E = 3.986004418e+5   #km^3/s^2, gravitational parameter for the Earth
R_E = 6371              #km, Earth radius

#PDC25 orbital elements for 2 body propagation from canvas
t0 = 61002.000                            #solution time, MJD TDB
t0_et = (t0 - 51544.5) * 86400            #converted to ephemeris time for use with spiceypy
q_PDC = 1.00581572e+00 * 1.496e+8         #km, perigee distance
e_PDC = 3.90397773e-01
i_PDC = 1.86515182e-01                    #radians
w_PDC = 3.51725315e-04
RAAN_PDC = 3.73976307e+00
a_PDC = q_PDC / (1 - e_PDC)
M0_PDC = 4.55537280e+00                   #PDC25 mean anomaly at time t0, rad

#Earth orbital elements at time t0 for 2-body propagation
q_E = 9.83287084e-01 * 1.496e+8           #km, perigee distance
e_E = 1.69558113e-02
i_E = 9.65537246e-05                      #radians
w_E = 5.10351418e+00
RAAN_E = 2.99846653e+00
M0_E = 5.57932001e+00 
a_E = q_E / (1 - e_E)

#Determining time until impact from orbital element solution time
t_impact = spice.str2et('2041-04-24T00:00:00.00Z')         #time of impact, et
t_2impact = (t_impact - t0_et)                             #time until impact in seconds
print('Time until impact:', t_2impact / (365.25 * 86400), 'years')
print('Solution time:', spice.et2utc(t0_et, 'C', 0))

#Convert orbital elements into r and v vectors at impact to determine Opik B plane coordinate basis
def rv(q, e, i, RAAN, w, M0, t0, t):
    (rx, ry, rz, vx, vy, vz) = spice.conics((q, e, i, RAAN, w, M0, t0, mu), t)
    r = np.array([rx, ry, rz])
    v = np.array([vx, vy, vz])
    return r, v

#Find position and velocity vectors for Earth anf PDC25 3 days before impact to solve for v_inf at SOI entry
r_PDC, v_PDC = rv(q_PDC, e_PDC, i_PDC, RAAN_PDC, w_PDC, M0_PDC, t0_et, t_impact - (3 * 86400))      #km, km/s
r_E, v_E = rv(q_E, e_E, i_E, RAAN_E, w_E, M0_E, t0_et, t_impact - (3 * 86400))                      #km, km/s

v_inf = v_E - v_PDC     #relative velocity of asteroid to Earth at SOI entry, v infinity
r_inf = r_E - r_PDC     #relative position of asteroid to Earth at SOI entry, r infinity
print('v_inf:', np.linalg.norm(v_inf), 'km/s')

#Find unit vector corresponding to v_PDC direction, this is equal to B-plane basis vector S.
S = v_PDC / np.linalg.norm(v_PDC)
print('B-plane basis vector S:', S)

#Compute xi and zeta from S and v_Earth
xi = np.cross(v_E, S) / np.linalg.norm(np.cross(v_E, S))
zeta = np.cross(-S, xi)
print('B-plane basis vector xi:', xi)
print('B-plane basis vector zeta:', zeta)

#Find B vector in order to solve for encounter point in the Opik plane
h_inf = np.cross(r_inf, v_inf)                #specific angular momentum of asteroid in its hyperbolic orbit w.r.t. Earth, NOT elliptic w.r.t. Sun
B = np.cross(S, h_inf) / np.linalg.norm(v_inf)

xi_enc = np.dot(B, xi)                        #xi coordinate of encounter point in Opik plane
zeta_enc = np.dot(B, zeta)                    #zeta coordinate of encounter point

#Compute the gravitational focusing radius of Earth
b_max = R_E * np.sqrt(1 + (2*mu_E/(R_E * np.linalg.norm(v_inf)**2)))          #gravitational focusing radius of Earth, au
print('Earth Inflated Radius (gravitational focusing):', b_max, 'km')


#Plotting
plt.figure(figsize=(10,8))
plt.scatter(xi_enc / R_E, zeta_enc / R_E, color='red', label='Encounter Point')
#Plot Earth as a circle with radius equal to the gravitational focusing radius
theta = np.linspace(0, 2*np.pi, 100)
x_E = (b_max / R_E) * np.cos(theta)
y_E = (b_max / R_E) * np.sin(theta)
plt.plot(x_E, y_E, color='blue', label='Gravitationally Focused Earth Radius (b_collision)')
plt.xlabel('xi (Earth Radii)')
plt.ylabel('zeta (Earth Radii)')
plt.legend(loc='upper left')
plt.title('Collision Depicted in the Opik B-plane')
plt.axis('equal')
plt.grid()
#plt.show()
#plt.savefig('Opik_B_plane.png', dpi=300)


#Perturbing the orbit via a slow continuous push method (such as Ion Beam Shepard or Gravity Tractor)
def Gauss_planetary(a, e, i, w, RAAN, f):
    global a_dr, a_dt, a_dh                                        #dr: radial, dt: theta trasverse, dh: out of plane
    theta = w + f
    r = a * (1 - e**2) / (1 + e*np.cos(f))
    h = np.sqrt(mu * a * (1 - e**2))
    p = a * (1 - e**2)
    T = 2 * np.pi * np.sqrt(a**3/mu)
    n = 2 * np.pi / T
    b = a * np.sqrt(1 - e**2)

    dRAAN = (r * np.sin(theta) / (h * np.sin(i))) * a_dh           #dRAAN/dt
    di = (r * np.cos(theta) / h) * a_dh                            #di/dt
    dw = (1 / (h*e)) * ((-p * np.cos(f) * a_dr) + ((p + r) * np.sin(f) * a_dt)) - ((r * np.sin(theta) * np.cos(i) / (h * np.sin(i))) * a_dh)
    da = (2 * a**2 / h) * ((e * np.sin(f) * a_dr) + (p/r * a_dt))
    de = (1/h) * ((p * np.sin(f) * a_dr) + ((p + r) * np.cos(f) + (r * e)) * a_dt)
    dM = n + ((b/(a*h*e)) * (((p * np.cos(f) - 2 * r * e) * a_dr) - ((p + r) * np.sin(f) * a_dt)))
    return da, de, di, dw, dRAAN, dM

def f_from_M(M, e):               #Using Kepler's equation and numerical integration (Newton's method) to convert anomalies in the other direction this time 
    E0 = M        #Initial guess for E
    g = 1         #Error metric
    itr = 0       #iteration counter
    tol = 1e-12   #tolerance

    while (abs(g) > tol) :
        g = E0 - e*np.sin(E0) - M
        dgdE = 1 - e*np.cos(E0)
        E1 = E0 - g/dgdE
        #print('iteration:',itr, 'g(E) = ',g, 'E1 = ', E1)
        E0 = E1
        itr = itr + 1
    E = E1
    f = 2 * np.atan2(np.tan(E/2) * np.sqrt(1 + e), np.sqrt(1 - e))
    if f < 0:
        f = 2*np.pi + f
    return f


#Input a perturbing acceleration and map the resulting encounter on the Opik B-plane. The following while loop slowly increases perturbing acceleration to find the minimum amount required
a_dr = 0
a_dt = 1e-12                  #km/s^2, small transverse acceleration
a_dh = 0

a = []
e = []
i = []

t_arr = spice.str2et('2030-09-17T00:00:00.00Z')       #time of arrival at asteroid (time slow deflection begins)
#Mean anomaly at arrival time t_arr
M0_PDC = M0_PDC + np.sqrt(mu/a_PDC**3) * (t_arr - t0_et)
M0_PDC = M0_PDC % (2*np.pi)                           #mean anomaly at arrival time, rad
print('Mean anomaly at arrival time:', M0_PDC, 'rad')

encounter_dist = 0           #value to initiate the loop
while encounter_dist < (R_E):
    f0_PDC = f_from_M(M0_PDC, e_PDC)

    time = np.arange(t_arr, t_impact, 10)             #propagate for 5 days after impact time to scan for a new MOID location
    dt = (t_impact - t_arr) / len(time)               #time step length, seconds
    print(dt)
    r_list = []
    v_list = []
    v_PDC_list = []
    v_E_list = []
    time_list = []
    for t in time:
        da, de, di, dw, dRAAN, dM = Gauss_planetary(a_PDC, e_PDC, i_PDC, w_PDC, RAAN_PDC, f0_PDC)
        a_PDC += da * dt
        e_PDC += de * dt
        i_PDC += di * dt
        w_PDC += dw * dt
        RAAN_PDC += dRAAN * dt
        M0_PDC += dM * dt                    #initial notation 0 meaning anomaly along trajectory before impact
        if M0_PDC < 0:
            M0_PDC = 2*np.pi + M0_PDC
        f0_PDC = f_from_M(M0_PDC, e_PDC)
        if t > t_impact - (5*86400):
            q_PDC = a_PDC * (1 - e_PDC)
            r_PDC, v_PDC = rv(q_PDC, e_PDC, i_PDC, RAAN_PDC, w_PDC, M0_PDC, t, t)        #computed from PDC elements at time t
            r_E, v_E = rv(q_E, e_E, i_E, RAAN_E, w_E, M0_E, t0_et, t)                    #computed from Earth elements at time t0
            r_rel = r_PDC - r_E                                                          #relative distance Earth to PDC
            v_rel = v_E - v_PDC                                                          #velocity of PDC relative to Earth
            time_list.append(t)
            r_list.append(r_rel)
            v_list.append(v_rel)
            v_PDC_list.append(v_PDC)
            v_E_list.append(v_E)

    #Now that the orbit was propagated past the initial MOID, find the perturbed encounter point by scanning the 10 timesteps surrounding near the original MOID
    dist = [np.linalg.norm(r) for r in r_list]
    min_dist = min(dist)
    min_index = dist.index(min_dist)
    min_dist_time = time_list[min_index]

    ##########Choose one of these based on the close approach time at which you'd like to consider v_infinity#################
    # v_PDC = v_PDC_list[min_index]                      #at close approach time
    # v_E = v_E_list[min_index]
    v_PDC = v_PDC_list[min_index - int(3*86400 / dt)]          #3 days before close approach time
    v_E = v_E_list[min_index - int(3*86400 / dt)]

    #Compute new encounter point in B-plane after perturbation
    S_deflect = v_PDC / np.linalg.norm(v_PDC)
    xi_deflect = np.cross(v_E, S_deflect) / np.linalg.norm(np.cross(v_E, S_deflect))
    zeta_deflect = np.cross(-S_deflect, xi_deflect)

    v_inf_deflect = v_list[min_index]
    r_inf_deflect = r_list[min_index]
    h_inf_deflect = np.cross(r_inf_deflect, v_inf_deflect)
    B_deflect = np.cross(S_deflect, h_inf_deflect) / np.linalg.norm(v_inf_deflect)

    xi_enc_deflect = np.dot(B_deflect, xi_deflect)                        #xi coordinate of encounter point in Opik plane
    zeta_enc_deflect = np.dot(B_deflect, zeta_deflect)                    #zeta coordinate of encounter point
    encounter_dist = np.linalg.norm([xi_enc_deflect, zeta_enc_deflect])

    a.append(a_PDC)
    e.append(e_PDC)
    i.append(i_PDC)

    #Reset for iterations where encounter distance is still less than 4 Earth radii
    a_dt += 1e-12
    q_PDC, e_PDC, i_PDC, w_PDC, RAAN_PDC, M0_PDC = (1.00581572e+00 * 1.496e+8, 3.90397773e-01, 1.86515182e-01, 3.51725315e-04, 3.73976307e+00, 6.2657863931842925)
    a_PDC = q_PDC/(1-e_PDC)
    

#Total perturbing acceleration needed
print('Perturbing acceleration: ', a_dt, 'km/s^2')
print('Encounter distance after perturbation: ', encounter_dist, 'km = ', encounter_dist / R_E, 'Earth Radii')
print('Time of minimum distance after perturbation: ', min_dist_time, 'seconds')
print("B magnitude:", np.linalg.norm(B_deflect))

#Compute the gravitational focusing radius of Earth
b_max_deflect = R_E * np.sqrt(1 + (2*mu_E/(R_E * np.linalg.norm(v_inf_deflect)**2)))
print('Earth Inflated Radius after deflection (gravitational focusing):', b_max_deflect, 'km')

#Total required delta V
delta_V = a_dt * (t_impact - t_arr)
print('Total required delta V: ', delta_V, 'km/s')

#Plotting
plt.figure(figsize=(10,8))
plt.scatter(xi_enc_deflect / R_E, zeta_enc_deflect / R_E, color='red', label='Encounter Point')
theta = np.linspace(0, 2*np.pi, 100)
x_E = (b_max_deflect / R_E) * np.cos(theta)
y_E = (b_max_deflect / R_E) * np.sin(theta)
plt.plot(x_E, y_E, color='blue', label='Gravitationally Focused Earth Radius (b_collision)')
plt.xlabel('xi (Earth Radii)')
plt.ylabel('zeta (Earth Radii)')
plt.legend(loc='upper left')
plt.title('Encounter After Slow Transverse Deflection')
plt.axis('equal')
plt.grid()
plt.show()
#plt.savefig('Opik_B_plane.png', dpi=300)