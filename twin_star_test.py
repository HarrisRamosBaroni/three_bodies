import numpy as np
from scipy.integrate import solve_ivp
import h5py

# Constants
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
AU = 1.496e11    # Astronomical unit in meters
solar_mass = 1.989e30  # Solar mass in kg
year = 3.154e7   # Seconds in a year

# System Parameters
m_A = 0.69 * solar_mass  # Mass of star A
m_B = 0.20 * solar_mass  # Mass of star B
m_planet = 1e24          # Planet mass (Earth-like planet)

# Binary star orbit parameters
a_binary = 0.224 * AU  # Semi-major axis of the binary (m)
e_binary = 0.2         # Binary eccentricity

# Planet orbit parameters
a_planet = 1.0 * AU  # Semi-major axis of the planet's orbit

# Initial Conditions
def initial_conditions():
    # Binary stars start at periapsis
    r_A = [-a_binary * (1 - e_binary) * m_B / (m_A + m_B), 0, 0]  # Star A
    r_B = [a_binary * (1 - e_binary) * m_A / (m_A + m_B), 0, 0]   # Star B

    # Binary star velocities at periapsis
    v_binary = np.sqrt(G * (m_A + m_B) * (1 + e_binary) / (a_binary * (1 - e_binary)))
    v_A = [0, v_binary * m_B / (m_A + m_B), 0]  # Star A velocity
    v_B = [0, -v_binary * m_A / (m_A + m_B), 0] # Star B velocity

    # Planet position relative to barycenter
    r_planet = [a_planet, 0, 0]

    # Planet velocity adjusted to account for binary barycenter motion
    v_planet_mag = np.sqrt(G * (m_A + m_B) / a_planet)
    v_planet = [0, v_planet_mag, 0]

    # Combine all positions and velocities
    y0 = np.concatenate([r_A, r_B, r_planet, v_A, v_B, v_planet])
    return y0

# Equations of motion
def equations_of_motion(t, y):
    # Extract positions
    r_A = y[:3]
    r_B = y[3:6]
    r_planet = y[6:9]

    # Extract velocities
    v_A = y[9:12]
    v_B = y[12:15]
    v_planet = y[15:18]

    # Relative distances
    r_AB = np.linalg.norm(r_A - r_B)
    r_Ap = np.linalg.norm(r_planet - r_A)
    r_Bp = np.linalg.norm(r_planet - r_B)

    # Accelerations
    a_A = -G * m_B * (r_A - r_B) / r_AB**3
    a_B = -G * m_A * (r_B - r_A) / r_AB**3
    a_planet = -G * m_A * (r_planet - r_A) / r_Ap**3 - G * m_B * (r_planet - r_B) / r_Bp**3

    # Combine accelerations and velocities
    a = np.concatenate([a_A, a_B, a_planet])
    v = np.concatenate([v_A, v_B, v_planet])
    return np.concatenate([v, a])

# Simulation parameters
t_span = (0, 5 * year)  # Simulate for 5 years
t_eval = np.linspace(*t_span, 3000)  # Time steps
y0 = initial_conditions()
solution = solve_ivp(equations_of_motion, t_span, y0, t_eval=t_eval, method='RK45')

# Check the shape of solution
print(solution.y.shape)  # Should print (18, len(t_eval))

# Extract results
r_A_sol = solution.y[:3]         # Star A position (3 rows, len(t_eval) columns)
r_B_sol = solution.y[3:6]         # Star B position (3 rows, len(t_eval) columns)
r_planet_sol = solution.y[6:9]    # Planet position (3 rows, len(t_eval) columns)
v_A_sol = solution.y[9:12]        # Star A velocity (3 rows, len(t_eval) columns)
v_B_sol = solution.y[12:15]       # Star B velocity (3 rows, len(t_eval) columns)
v_planet_sol = solution.y[15:18]  # Planet velocity (3 rows, len(t_eval) columns)

# Debugging: Check the size of each result array
print(f"r_A_sol shape: {r_A_sol.shape}")
print(f"r_B_sol shape: {r_B_sol.shape}")
print(f"r_planet_sol shape: {r_planet_sol.shape}")

# Derived quantities
kinetic_energy = []
potential_energy = []
total_energy = []
momentum = []
angular_momentum = []

# Loop over all the time steps
for i in range(len(t_eval)):  # i goes from 0 to len(t_eval) - 1
    # Extract positions and velocities at time step i
    r_A = r_A_sol[:, i]           # Star A position at time i
    r_B = r_B_sol[:, i]           # Star B position at time i
    r_planet = r_planet_sol[:, i] # Planet position at time i
    v_A = v_A_sol[:, i]           # Star A velocity at time i
    v_B = v_B_sol[:, i]           # Star B velocity at time i
    v_planet = v_planet_sol[:, i] # Planet velocity at time i

    # Kinetic energy
    KE_A = 0.5 * m_A * np.dot(v_A, v_A)
    KE_B = 0.5 * m_B * np.dot(v_B, v_B)
    KE_planet = 0.5 * m_planet * np.dot(v_planet, v_planet)
    KE_total = KE_A + KE_B + KE_planet
    kinetic_energy.append(KE_total)

    # Potential energy
    r_AB = np.linalg.norm(r_A - r_B)
    r_Ap = np.linalg.norm(r_planet - r_A)
    r_Bp = np.linalg.norm(r_planet - r_B)
    PE_AB = -G * m_A * m_B / r_AB
    PE_Ap = -G * m_A * m_planet / r_Ap
    PE_Bp = -G * m_B * m_planet / r_Bp
    PE_total = PE_AB + PE_Ap + PE_Bp
    potential_energy.append(PE_total)

    # Total energy
    total_energy.append(KE_total + PE_total)

    # Momentum
    P_A = m_A * v_A
    P_B = m_B * v_B
    P_planet = m_planet * v_planet
    momentum.append(P_A + P_B + P_planet)

    # Angular momentum
    L_A = np.cross(r_A, P_A)
    L_B = np.cross(r_B, P_B)
    L_planet = np.cross(r_planet, P_planet)
    angular_momentum.append(L_A + L_B + L_planet)

# Save data
with h5py.File("simulation_data.hdf5", "w") as f:
    f.create_dataset("time", data=t_eval)
    f.create_dataset("positions_A", data=r_A_sol)
    f.create_dataset("positions_B", data=r_B_sol)
    f.create_dataset("positions_planet", data=r_planet_sol)
    f.create_dataset("velocities_A", data=v_A_sol)
    f.create_dataset("velocities_B", data=v_B_sol)
    f.create_dataset("velocities_planet", data=v_planet_sol)
    f.create_dataset("kinetic_energy", data=kinetic_energy)
    f.create_dataset("potential_energy", data=potential_energy)
    f.create_dataset("total_energy", data=total_energy)
    f.create_dataset("momentum", data=momentum)
    f.create_dataset("angular_momentum", data=angular_momentum)

print("Simulation data saved to simulation_data.hdf5")
