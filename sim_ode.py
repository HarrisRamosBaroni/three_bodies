import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from astroquery.jplhorizons import Horizons
import h5py

#-------------------
# important stuff
#-------------------

# Constants
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
c = 3e8  # Speed of light (m/s)
AU = 1.496e11    # Astronomical unit in meters
solar_mass = 1.989e30  # Solar mass in kg
year = 3.154e7   # Seconds in a year

# System Parameters
m_A = 1 * solar_mass  # Mass of body A
m_B = 0.00189813 * solar_mass  # Mass of body B
m_C = 0.000578 * solar_mass  # Mass of body C
masses = (m_A, m_B, m_C)


#-------------------
# random initial conditions
# #-------------------

# # Masses (in kg)
# m_A = 2.188e30  # Alpha Centauri A
# m_B = 1.81e30   # Alpha Centauri B
# m_C = 0.2429e30  # Proxima Centauri
# masses = (m_A, m_B, m_C)

# # Semi-major axis (in meters)
# # a_AB = 3.5455e12  # Approximate semi-major axis of Alpha Centauri A and B (23.7 au)
# a_AB = 35.6*AU  # Approximate semi-major axis of Alpha Centauri A and B (23.7 au)

# # Gravitational parameter for Alpha Centauri A and B system
# mu_AB = G * (m_A + m_B)

# # Initial positions and velocities
# r_A = np.array([-a_AB * (m_B / (m_A + m_B)), 0, 0])  # Position of A
# r_B = np.array([a_AB * (m_A / (m_A + m_B)), 0, 0])   # Position of B

# # Calculate the orbital velocities using the Vis-Viva equation
# # v_A = np.array([0, np.sqrt(mu_AB * (2/np.linalg.norm(r_A) - 1/a_AB)), 0])  # Velocity of A
# # v_B = np.array([0, -np.sqrt(mu_AB * (2/np.linalg.norm(r_B) - 1/a_AB)), 0])  # Velocity of B
# # v_A = np.array([0, np.sqrt(mu_AB * (1/a_AB)), 0])  # Velocity of A
# # v_B = np.array([0, -np.sqrt(mu_AB * (1/a_AB)), 0])  # Velocity of B
# v_A = np.array([0, 4.8e3, 0])  # Velocity of A
# v_B = np.array([0, -4.8e3, 0])  # Velocity of B

# # Position and velocity of Proxima Centauri (C) (still treated as orbiting the center of mass)
# a_C = 1.3015e15  # Semi-major axis of Proxima Centauri's orbit (8700 au)
# r_C = np.array([a_C, 0, 0])  # Position of C, assuming it lies along x-axis
# v_C = np.array([0, np.sqrt(G * (m_A + m_B) / np.linalg.norm(r_C)), 0])  # Velocity of C







#-------------------
# Obtain real data
#-------------------
def obtain_ephemeris(bodies_names, bodies_ids, start_date='2024-01-01', end_date='2024-12-31', step='1d'):
    """
    Obtain ephemeris data (positions and velocities) for the given celestial bodies.
    
    Parameters:
    - bodies_names: List of names of the bodies (e.g., ['Jupiter', 'Sun', 'Earth', 'Saturn'])
    - bodies_ids: List of JPL Horizons IDs corresponding to the bodies (e.g., ['599', '10', '399', '699'])
    - start_date: Start date for the ephemeris query (default '2024-01-01')
    - end_date: End date for the ephemeris query (default '2024-12-31')
    - step: Step size for the query (default '1d')
    
    Returns:
    - data: Dictionary containing time, positions, and velocities for all bodies
    """
    if len(bodies_names) != len(bodies_ids):
        raise ValueError("The number of body names must match the number of body IDs.")
    
    trajectories = {}
    time_stamps = None  # Placeholder for time stamps
    
    # Query ephemeris data for each body
    for body_name, body_id in zip(bodies_names, bodies_ids):
        obj = Horizons(id=body_id, location='@0', 
                       epochs={'start': start_date, 'stop': end_date, 'step': step})
        vecs = obj.vectors()
        
        # Store time stamps only once (assume all bodies have the same time range)
        if time_stamps is None:
            time_stamps = vecs['datetime_jd']
        
        # Store positions and velocities for each body
        trajectories[body_name] = {
            'x': vecs['x'], 'y': vecs['y'], 'z': vecs['z'],
            'vx': vecs['vx'], 'vy': vecs['vy'], 'vz': vecs['vz']
        }
    
    # Organize the data into a dictionary with human-readable labels
    data = {'time': time_stamps}
    for body_name in bodies_names:
        data[body_name] = trajectories[body_name]

    return data

#---------------------------------------
# Initial conditions and sim parameters
#---------------------------------------
def initial_conditions(data, bodies_names):
    positions = []
    velocities = []
    
    for body in bodies_names:
        # Extract positions and velocities for the body
        positions.extend([data[body]['x'][0], data[body]['y'][0], data[body]['z'][0]])
        velocities.extend([data[body]['vx'][0], data[body]['vy'][0], data[body]['vz'][0]])
    
    # Concatenate positions and velocities
    y0 = np.array(positions + velocities)
    return y0

# Define t_span and t_eval based on data['time']
def time_parameters(data):
    time_jd = np.array(data['time'])  # Julian dates
    t_span = (time_jd[0], time_jd[-1])  # Start and end times for the ODE solver
    t_eval = time_jd  # Times at which the solution is evaluated
    return t_span, t_eval

#-------------
# ODE stuff
#-------------
def eih_accelerations(t, state, masses, G, c):
    '''
    Einstein-Infeld-Hoffmann (EIH) equations of motion
    '''
    n = len(masses)
    x = state[:3*n].reshape(n, 3)
    v = state[3*n:].reshape(n, 3)
    a = np.zeros_like(x)

    for i in range(n):
        for j in range(n):
            if i != j:
                r_ij = x[j] - x[i]
                r_ij_mag = np.linalg.norm(r_ij)
                r_ij_unit = r_ij / r_ij_mag
                v_ij = v[j] - v[i]

                # Newtonian term
                a[i] += G * masses[j] * r_ij_unit / r_ij_mag**2

                # 1PN corrections
                v_i_sq = np.dot(v[i], v[i])
                v_j_sq = np.dot(v[j], v[j])
                v_ij_sq = np.dot(v_ij, v_ij)

                # EIH terms
                a[i] += G * masses[j] / (c**2 * r_ij_mag**2) * (
                    r_ij_unit * (
                        4 * G * (masses[i] + masses[j]) / r_ij_mag
                        - v_i_sq - 2 * v_j_sq + 4 * np.dot(v[i], v[j])
                        - 1.5 * v_ij_sq
                    )
                    + 4 * v_ij * np.dot(v_ij, r_ij_unit)
                )

    return np.concatenate([v.flatten(), a.flatten()])

def calculate_gravitational_force(pos_1, pos_2, mass_1, mass_2):
    '''
    Function to compute gravitational force between two bodies
    '''
    distance_vector = pos_2 - pos_1
    distance = np.linalg.norm(distance_vector)
    if distance == 0:  # Prevent division by zero in case of overlapping bodies
        return np.zeros_like(distance_vector)
    force = G * mass_1 * mass_2 * distance_vector / distance**3
    return force

def newton_accelerations(t, y, masses):
    '''
    Equations of motion of n bodies (newtonian)
    '''
    n = len(masses)  # no. of bodies
    
    # Unpack positions and velocities from the state vector y
    positions = y[:n*3].reshape((n, 3))   # (n x 3) array of positions
    velocities = y[n*3:].reshape((n, 3))  # (n x 3) array of velocities
    
    accelerations = np.zeros_like(positions)  # initialise accel.s (n x 3 array)

    # Compute the gravitational forces between all pairs of bodies
    for i in range(n):
        total_force = np.zeros(3)  # total force on body i
        for j in range(n):
            if i != j:
                force_ij = calculate_gravitational_force(positions[i], positions[j], masses[i], masses[j])
                total_force += force_ij
        accelerations[i] = total_force / masses[i]  # newtons second law F = ma

    # Return the derivatives of the positions and velocities
    return np.concatenate([velocities.flatten(), accelerations.flatten()])

# Obtain eph data
# bodies_names = ['Jupiter', 'Sun', 'Earth', 'Saturn']
# bodies_ids = ['599', '10', '399', '699']
bodies_names = ['Sun', 'Jupiter', 'Saturn']
bodies_ids = ['10', '599', '699']
ephemeris_data = obtain_ephemeris(bodies_names, bodies_ids)

# Simulation parameters
# y0 = np.concatenate([r_A, r_B, r_C, v_A, v_B, v_C])
y0 = initial_conditions(ephemeris_data, bodies_names)
# print(y0)
# print(y0.shape)  # (18,)
t_span = (0, 5 * year)  # Simulate for 5 years
t_eval = np.linspace(*t_span, 2000)  # Ensure consistent steps. make many steps, eg 600 in 5 years is not good enough and will get # solution.message = 'Required step size is less than spacing between numbers.'
# t_span, t_eval = time_parameters(ephemeris_data)
# print(t_span, t_eval)

# Solve the system
solution = solve_ivp(newton_accelerations, t_span, y0, t_eval=t_eval, method='RK45', args=(masses,))
# solution = solve_ivp(eih_accelerations, t_span, y0, t_eval=t_eval, method='RK45', args=(masses, G, c))

# Extract results with consistent time steps
t_result = solution.t
# print(solution.y)
soly_num = np.array(solution.y)
print(soly_num.shape)
positions = solution.y[:9, :].T  # Positions: r_A, r_B, r_C
velocities = solution.y[9:, :].T  # Velocities: v_A, v_B, v_C

#------------------------------
# Sim data save
#------------------------------
# Create a pandas DataFrame
data = {
    'time': t_result,
    'r_A_x': positions[:, 0], 'r_A_y': positions[:, 1], 'r_A_z': positions[:, 2],
    'r_B_x': positions[:, 3], 'r_B_y': positions[:, 4], 'r_B_z': positions[:, 5],
    'r_C_x': positions[:, 6], 'r_C_y': positions[:, 7], 'r_C_z': positions[:, 8],
    'v_A_x': velocities[:, 0], 'v_A_y': velocities[:, 1], 'v_A_z': velocities[:, 2],
    'v_B_x': velocities[:, 3], 'v_B_y': velocities[:, 4], 'v_B_z': velocities[:, 5],
    'v_C_x': velocities[:, 6], 'v_C_y': velocities[:, 7], 'v_C_z': velocities[:, 8],
}
# data = {
#     'time': t_result,  # Include time if needed
#     'A': {
#         'x': positions[:, 0], 'y': positions[:, 1], 'z': positions[:, 2],
#         'vx': velocities[:, 0], 'vy': velocities[:, 1], 'vz': velocities[:, 2]
#     },
#     'B': {
#         'x': positions[:, 3], 'y': positions[:, 4], 'z': positions[:, 5],
#         'vx': velocities[:, 3], 'vy': velocities[:, 4], 'vz': velocities[:, 5]
#     },
#     'C': {
#         'x': positions[:, 6], 'y': positions[:, 7], 'z': positions[:, 8],
#         'vx': velocities[:, 6], 'vy': velocities[:, 7], 'vz': velocities[:, 8]
#     }
# }
df = pd.DataFrame(data)

# Ask the user if they want to save the data
save_data = input("Do you want to save the simulation data? (y/n): ").strip().lower()
if save_data == 'y':
    # Save results to HDF5 file
    df.to_csv('simulation_data.csv', mode='w')
    print("Simulation data saved to simulation_data.hdf5")
else:
    print("Simulation data not saved.")
    



#------------------------------
# Ephemeris data save
#------------------------------
# Assuming ephemeris_data is the output from the obtain_ephemeris function
time = ephemeris_data['time']

# Assume the body names are 'Earth', 'Jupiter', 'Saturn' and map them to 'A', 'B', 'C'
positions = []
velocities = []

# Extract positions and velocities for the three bodies
for body_name in bodies_names:
    body_data = ephemeris_data[body_name]
    
    # Extract position (x, y, z) and velocity (vx, vy, vz) for each body
    body_positions = np.array([body_data['x'], body_data['y'], body_data['z']]).T
    body_velocities = np.array([body_data['vx'], body_data['vy'], body_data['vz']]).T
    
    positions.append(body_positions)
    velocities.append(body_velocities)

# Stack all positions and velocities horizontally (shape will be (n, 9) for 3 bodies)
positions = np.hstack(positions)
velocities = np.hstack(velocities)

# Now create the data dictionary in the desired format
ephemeris_data = {
    'time': time,
    'r_A_x': positions[:, 0], 'r_A_y': positions[:, 1], 'r_A_z': positions[:, 2],
    'r_B_x': positions[:, 3], 'r_B_y': positions[:, 4], 'r_B_z': positions[:, 5],
    'r_C_x': positions[:, 6], 'r_C_y': positions[:, 7], 'r_C_z': positions[:, 8],
    'v_A_x': velocities[:, 0], 'v_A_y': velocities[:, 1], 'v_A_z': velocities[:, 2],
    'v_B_x': velocities[:, 3], 'v_B_y': velocities[:, 4], 'v_B_z': velocities[:, 5],
    'v_C_x': velocities[:, 6], 'v_C_y': velocities[:, 7], 'v_C_z': velocities[:, 8],
}
df = pd.DataFrame(ephemeris_data)

# Ask the user if they want to save the data
save_data = input("Do you want to save the ephemeris data? (y/n): ").strip().lower()
if save_data == 'y':
    # Save results to HDF5 file
    df.to_csv('ephemeris_data.csv', mode='w')
    print("Ephemeris data saved to ephemeris_data.hdf5")
else:
    print("Ephemeris data not saved.")
