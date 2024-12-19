import numpy as np
import pandas as pd
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
    r_A = [-a_binary * (1 - e_binary) * m_B / (m_A + m_B), 0, 0]  # Star A
    r_B = [a_binary * (1 - e_binary) * m_A / (m_A + m_B), 0, 0]   # Star B
    v_binary = np.sqrt(G * (m_A + m_B) * (1 + e_binary) / (a_binary * (1 - e_binary)))
    v_A = [0, v_binary * m_B / (m_A + m_B), 0]  # Star A velocity
    v_B = [0, -v_binary * m_A / (m_A + m_B), 0] # Star B velocity
    r_planet = [a_planet, 0, 0]
    v_planet_mag = np.sqrt(G * (m_A + m_B) / a_planet)
    v_planet = [0, v_planet_mag, 0]
    y0 = np.concatenate([r_A, r_B, r_planet, v_A, v_B, v_planet])
    return y0

# Equations of motion
def equations_of_motion(t, y):
    r_A, r_B, r_planet = y[:3], y[3:6], y[6:9]
    v_A, v_B, v_planet = y[9:12], y[12:15], y[15:18]
    r_AB = np.linalg.norm(r_A - r_B)
    r_Ap = np.linalg.norm(r_planet - r_A)
    r_Bp = np.linalg.norm(r_planet - r_B)
    a_A = -G * m_B * (r_A - r_B) / r_AB**3
    a_B = -G * m_A * (r_B - r_A) / r_AB**3
    a_planet = (-G * m_A * (r_planet - r_A) / r_Ap**3
                -G * m_B * (r_planet - r_B) / r_Bp**3)
    a = np.concatenate([a_A, a_B, a_planet])
    v = np.concatenate([v_A, v_B, v_planet])
    return np.concatenate([v, a])

# Simulation parameters
t_span = (0, 5 * year)  # Simulate for 5 years
t_eval = np.linspace(*t_span, 600)  # Ensure consistent steps
y0 = initial_conditions()

# Solve the system
solution = solve_ivp(equations_of_motion, t_span, y0, t_eval=t_eval, method='RK45')

# Extract results with consistent time steps
t_result = solution.t
positions = solution.y[:9, :].T  # Positions: r_A, r_B, r_planet
velocities = solution.y[9:, :].T  # Velocities: v_A, v_B, v_planet

# Create a pandas DataFrame
data = {
    'time': t_result,
    'r_A_x': positions[:, 0], 'r_A_y': positions[:, 1], 'r_A_z': positions[:, 2],
    'r_B_x': positions[:, 3], 'r_B_y': positions[:, 4], 'r_B_z': positions[:, 5],
    'r_planet_x': positions[:, 6], 'r_planet_y': positions[:, 7], 'r_planet_z': positions[:, 8],
    'v_A_x': velocities[:, 0], 'v_A_y': velocities[:, 1], 'v_A_z': velocities[:, 2],
    'v_B_x': velocities[:, 3], 'v_B_y': velocities[:, 4], 'v_B_z': velocities[:, 5],
    'v_planet_x': velocities[:, 6], 'v_planet_y': velocities[:, 7], 'v_planet_z': velocities[:, 8],
}
df = pd.DataFrame(data)

# Ask the user if they want to save the data
save_data = input("Do you want to save the simulation data? (y/n): ").strip().lower()

if save_data == 'y':
    # Save results to HDF5 file
    df.to_hdf('simulation_data.hdf5', key='df', mode='w')
    print("Simulation data saved to simulation_data.hdf5")
else:
    print("Simulation data not saved.")