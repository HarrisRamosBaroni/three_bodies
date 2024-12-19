import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from astroquery.jplhorizons import Horizons
import astropy.units as u
import h5py
import os
from datetime import datetime

#-------------------
# Constants
#-------------------
G = 6.6743015e-11  # Gravitational constant (m^3 kg^-1 s^-2). Relative standard uncertainty is 2.2e−5. CODATA-recommended value of the gravitational constant.
c = 2.99792458e8   # Speed of light (m/s). This value is EXACT by definition.
solar_mass = 1.988475e30   # Solar mass in kg (1.988475±0.000092 e30) Prša, Andrej; ...; Donald W.; Laskar, Jacques (2016-08-01). "NOMINAL VALUES FOR SELECTED SOLAR AND PLANETARY QUANTITIES: IAU 2015 RESOLUTION B3 * †"
jupiter_mass = 1.89813e27  # Mass of jupiter in kg
earth_mass = 5.9722e24     # Mass of earth in kg (5.9722±0.0006 e24). https://en.wikipedia.org/wiki/Earth_mass
year = 3.154e7   # Seconds in a year

#-------------------
# Obtain real data
#-------------------
def obtain_ephemeris(bodies_names, bodies_ids, start_date='2022-01-01', end_date='2023-12-31', step='1d'):
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
        obj = Horizons(id=body_id, location='500@0', 
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
    
    # Organise the data into a dictionary
    data = {'time': time_stamps}
    for body_name in bodies_names:
        data[body_name] = trajectories[body_name]

    return data

#---------------------------------------
# Initial conditions and sim parameters
#---------------------------------------
def initial_conditions_si(data, bodies_names, noise_std_pos=1e3, noise_std_vel=1e-2):
    """
    Obtain initial conditions with added noise for Monte Carlo simulations.
    
    Parameters:
    - data: dictionary containing ephemeris data
    - bodies_names: names of celestial bodies (list of 3 strings)
    - noise_std_position: standard deviation of Gaussian noise to add to positions (in meters)
    - noise_std_velocity: standard deviation of Gaussian noise to add to velocities (in m/s)
    
    Returns:
    - y0: initial values for the IVP with noise [x, y, z, vx, vy, vz]
    """
    positions = []
    velocities = []
    
    for body in bodies_names:
        # Convert positions and velocities using astropy units
        # Assuming the input data is in AU and AU/day, we will convert them to meters and meters per second
        
        # Position conversion: AU to meters
        x = (data[body]['x'][0] * u.au).to(u.m).value
        y = (data[body]['y'][0] * u.au).to(u.m).value
        z = (data[body]['z'][0] * u.au).to(u.m).value
        
        # Velocity conversion: AU/day to m/s
        vx = (data[body]['vx'][0] * u.au / u.day).to(u.m / u.s).value
        vy = (data[body]['vy'][0] * u.au / u.day).to(u.m / u.s).value
        vz = (data[body]['vz'][0] * u.au / u.day).to(u.m / u.s).value
        
        # Add Gaussian noise to positions and velocities
        x += np.random.normal(0, noise_std_pos)
        y += np.random.normal(0, noise_std_pos)
        z += np.random.normal(0, noise_std_pos)
        vx += np.random.normal(0, noise_std_vel)
        vy += np.random.normal(0, noise_std_vel)
        vz += np.random.normal(0, noise_std_vel)
        
        # Append converted values to the respective lists
        positions.extend([x, y, z])
        velocities.extend([vx, vy, vz])
    
    # Concatenate positions and velocities to form the initial state vector
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
def eih_accelerations(t, state, masses): # , G=G, c=c):
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
    epsilon = 1e-10  # Small threshold to avoid instability in division
    if distance < epsilon:  # If distance is too small, treat it as 0
        return np.zeros_like(distance_vector)
    force = G * mass_1 * mass_2 * distance_vector / distance**3
    return force

def newton_accelerations(t, y, masses):
    '''
    Equations of motion of n bodies (newtonian)
    '''
    n = len(masses)  # no. of bodies
    
    # Unpack positions and velocities from the state vector y
    positions = y[:n*3].reshape(n, 3)  # (n x 3) array of positions
    velocities = y[n*3:].reshape(n, 3) # (n x 3) array of velocities
    
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

class Sim():
    def __init__(self, bodies_names, bodies_ids, start_date, end_date, ode, noise_std_pos=0.0, noise_std_vel=0.0, method='RK45', debug_prints=True):
        self.bodies_names = bodies_names  # names of the celestial bodies of interest (for labelling purposes)
        self.bodies_ids = bodies_ids      # identification number of the celestial bodies of interest
        self.start_date = start_date      # ephemeris start date eg '2022-01-01'
        self.end_date = end_date          # ephemeris end date
        # self.noise_percentage = noise_percentage
        self.noise_std_pos = noise_std_pos 
        self.noise_std_vel = noise_std_vel
        self.method = method              # initial value problem solver method. 'RK45', 'LSODA', ...
        self.ode = ode                    # function describing system ode. eih_accelerations or newton_accelerations 
        self.debug_prints = debug_prints

        self.ephemeris_data = obtain_ephemeris(bodies_names, bodies_ids, start_date=self.start_date, end_date=self.end_date, step='1h')
        # Simulation parameters
        # self.y0 = initial_conditions_si(self.ephemeris_data, bodies_names, noise_std_pos, noise_std_vel)
        # self.t_span, self.t_eval = time_parameters(self.ephemeris_data)
        sampled_hours = len(self.ephemeris_data['time'])
        self.t_span = (0, 3600 * sampled_hours)  # run sim for 3600 * sampled_hours seconds
        self.t_eval = np.linspace(*self.t_span, sampled_hours)  # steps every hour
        if self.debug_prints:
            # print("type(self.ephemeris_data['time'])", type(self.ephemeris_data['time']))
            unmasked_length = self.ephemeris_data['time'].count()
            print("unmasked_length", unmasked_length)
            print("len(self.ephemeris_data['time'])", len(self.ephemeris_data['time']))
            # print("self.y0", self.y0)  # initial conditions
            # print(y0.shape)  # (18,)
            # print(t_span, t_eval)

        self._y0 = None  # Private attribute to hold the value of y0
    
    @property
    def y0(self):
        """Getter for y0."""
        return self._y0    

    @y0.setter
    def y0(self, dummy):
        """Setter function for y0 that calculates initial conditions."""
        self._y0 = initial_conditions_si(self.ephemeris_data, self.bodies_names, self.noise_std_pos, self.noise_std_vel)
    
    def solve(self):
        # Solve the system
        solution = solve_ivp(self.ode, self.t_span, self.y0, t_eval=self.t_eval, method=self.method, args=(masses,), rtol=1e-8, atol=1e-8)

        # Extract results with consistent time steps
        self.t_result = solution.t
        if self.debug_prints:
            soly_num = np.array(solution.y)
            print("solution.y.shape", soly_num.shape)
            # print(solution.y)
        self.positions = solution.y[:9, :].T  # Positions: r_A, r_B, r_C
        self.velocities = solution.y[9:, :].T  # Velocities: v_A, v_B, v_C

    @staticmethod
    def get_ode_directory(ode_function):
        # Function to get the directory path based on ODE name
        ode_name = ode_function.__name__  # Get the function name (e.g., "newton_accelerations")
        return os.path.join('data', ode_name)

    @staticmethod
    def format_body_names(body_names):
        # Function to format the body names into a string (e.g., "Sun-Earth-Moon")
        return '-'.join(body_names)
    
    def save_sim_data(self):
        #------------------------------
        # Sim data save
        #------------------------------
        # Create a pandas DataFrame
        t_result = self.t_result
        positions= self.positions
        velocities = self.velocities

        data = {
            'time': t_result,
            'r_A_x': positions[:, 0], 'r_A_y': positions[:, 1], 'r_A_z': positions[:, 2],
            'r_B_x': positions[:, 3], 'r_B_y': positions[:, 4], 'r_B_z': positions[:, 5],
            'r_C_x': positions[:, 6], 'r_C_y': positions[:, 7], 'r_C_z': positions[:, 8],
            'v_A_x': velocities[:, 0], 'v_A_y': velocities[:, 1], 'v_A_z': velocities[:, 2],
            'v_B_x': velocities[:, 3], 'v_B_y': velocities[:, 4], 'v_B_z': velocities[:, 5],
            'v_C_x': velocities[:, 6], 'v_C_y': velocities[:, 7], 'v_C_z': velocities[:, 8],
        }
        df = pd.DataFrame(data)

        # Save the simulation data
        save_data = input("Do you want to save the simulation data? (y/n): ").strip().lower()
        if save_data == 'y':
            # Get the ODE function name and corresponding directory
            # print("self.ode", self.ode)
            ode_directory = self.get_ode_directory(self.ode)
            os.makedirs(ode_directory, exist_ok=True)  # Ensure the directory exists (create if not existing already)
            
            # Format the body names into a string
            body_names_str = self.format_body_names(self.bodies_names)
            
            # Generate the filename with the required format
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{body_names_str}_{start_date}_{end_date}_noise{int(self.noise_std_pos)}_{int(self.noise_std_vel)}-{timestamp}.csv"
            file_path = os.path.join(ode_directory, filename)

            # Save the data to the CSV file
            df.to_csv(file_path, mode='w')
            print(f"Simulation data saved to {file_path}")
        else:
            print("Simulation data not saved.")            

    def save_eph_data(self):
        #------------------------------
        # Ephemeris data save
        #------------------------------
        time = self.ephemeris_data['time']

        positions = []
        velocities = []

        # Extract positions and velocities for the three bodies
        for body_name in bodies_names:
            body_data = self.ephemeris_data[body_name]
            
            # Extract position (x, y, z) and velocity (vx, vy, vz) for each body
            body_positions = np.array([body_data['x'], body_data['y'], body_data['z']]).T
            body_velocities = np.array([body_data['vx'], body_data['vy'], body_data['vz']]).T
            
            positions.append(body_positions)
            velocities.append(body_velocities)

        # Stack all positions and velocities horizontally. (shape will be (n, 9) for 3 bodies)
        positions = np.hstack(positions)
        velocities = np.hstack(velocities)

        # Create the data dictionary in the same format as the sim data
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

        # Save Ephemeris Data (same directory structure idea)
        save_ephemeris_data = input("Do you want to save the ephemeris data? (y/n): ").strip().lower()
        if save_ephemeris_data == 'y':
            # Ensure ephemeris data is saved in a subdirectory
            ephemeris_directory = os.path.join('data', 'ephemeris')
            os.makedirs(ephemeris_directory, exist_ok=True) # Ensure the directory exists (create if not existing already)
            
            # Format the body names into a string
            body_names_str = self.format_body_names(self.bodies_names)
            
            # Generate the filename with the required format
            ephemeris_filename = f"{body_names_str}_{self.start_date}_{self.end_date}.csv"
            ephemeris_file_path = os.path.join(ephemeris_directory, ephemeris_filename)

            # Save the ephemeris data to the CSV file
            df.to_csv(ephemeris_file_path, mode='w')
            print(f"Ephemeris data saved to {ephemeris_file_path}")
        else:
            print("Ephemeris data not saved.")        

#------------------------------
# System parameters: masses
#------------------------------
'''
"A new set of "current best estimates" for various astronomical constants was approved
the 27th General Assembly of the International Astronomical Union (IAU) in August 2009."
https://en.wikipedia.org/wiki/Planetary_mass
'''
# m_A = 1 * solar_mass  # Mass of body A
# m_B = 954.791915e-6 * solar_mass  # Mass of jupiter and satellites (954.7919e-6 Msol without satellites) Jacobson, R. A.; Haw, R. J.; McElrath, T. P.; Antreasian, P. G. (2000).
# m_C = 285.8856708e-6 * solar_mass  # Mass of saturn and satellites (285.885670e-6 Msol without satellites) Jacobson, R. A.; Antreasian, P. G.; Bordi, J. J.; Criddle, K. E.; et al. (2006).

# m_A = 1047.348644 * jupiter_mass
# m_B = jupiter_mass  # Mass of jupiter alone
# m_C = 0.29942197 * jupiter_mass # Mass of saturn alone

m_A = 332946.0487 * earth_mass  # (332,946.0487±0.0007). The cited value is the recommended value published by the International Astronomical Union in 2009.
m_B = earth_mass  # Mass of earth alone
m_C = 0.0123000371 * earth_mass # Mass of moon alone. Pitjeva, E.V.; Standish, E.M. (1 April 2009).
masses = (m_A, m_B, m_C)

# Obtain eph data
# bodies_names = ['Jupiter', 'Sun', 'Earth', 'Saturn']
# bodies_ids = ['599', '10', '399', '699']
bodies_names = ['Sun', 'Earth', 'Moon']
bodies_ids = ['10', '399', '301']
# bodies_names = ['Sun', 'Jupiter barycenter', 'Saturn barycenter']
# bodies_ids = ['10', '5', '6']
#ephemeris_data = obtain_ephemeris(bodies_names, bodies_ids , step='1h')
start_date='2022-01-01'
end_date='2023-12-31'

sim = Sim(bodies_names=bodies_names, bodies_ids=bodies_ids, start_date=start_date, end_date=end_date,
          ode=newton_accelerations, noise_std_pos=1e3, noise_std_vel=0.0, method='RK45')
sim.save_eph_data()
n_runs = 3
for _ in range(n_runs):
    sim.y0 = None  # obtain initial condition with new gaussian noise
    print("sim.y0", sim.y0)
    sim.solve()
    sim.save_sim_data()
