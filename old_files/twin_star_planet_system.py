import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Constants
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
c = 3e8  # Speed of light (m/s)
au = 1.496e+11  # metres in au

# Masses (in kg)
m_A = 2.188e30  # Alpha Centauri A
m_B = 1.81e30   # Alpha Centauri B
m_C = 0.2429e30  # Proxima Centauri
masses = (m_A, m_B, m_C)

# Semi-major axis (in meters)
# a_AB = 3.5455e12  # Approximate semi-major axis of Alpha Centauri A and B (23.7 au)
a_AB = 35.6*au  # Approximate semi-major axis of Alpha Centauri A and B (23.7 au)

# Gravitational parameter for Alpha Centauri A and B system
mu_AB = G * (m_A + m_B)

# Initial positions and velocities
r_A = np.array([-a_AB * (m_B / (m_A + m_B)), 0, 0])  # Position of A
r_B = np.array([a_AB * (m_A / (m_A + m_B)), 0, 0])   # Position of B

# Calculate the orbital velocities using the Vis-Viva equation
# v_A = np.array([0, np.sqrt(mu_AB * (2/np.linalg.norm(r_A) - 1/a_AB)), 0])  # Velocity of A
# v_B = np.array([0, -np.sqrt(mu_AB * (2/np.linalg.norm(r_B) - 1/a_AB)), 0])  # Velocity of B
# v_A = np.array([0, np.sqrt(mu_AB * (1/a_AB)), 0])  # Velocity of A
# v_B = np.array([0, -np.sqrt(mu_AB * (1/a_AB)), 0])  # Velocity of B
v_A = np.array([0, 4.8e3, 0])  # Velocity of A
v_B = np.array([0, -4.8e3, 0])  # Velocity of B

# Position and velocity of Proxima Centauri (C) (still treated as orbiting the center of mass)
a_C = 1.3015e15  # Semi-major axis of Proxima Centauri's orbit (8700 au)
r_C = np.array([a_C, 0, 0])  # Position of C, assuming it lies along x-axis
v_C = np.array([0, np.sqrt(G * (m_A + m_B) / np.linalg.norm(r_C)), 0])  # Velocity of C

# Print the initial conditions
print("Initial Conditions:")
print(f"Position of A: {r_A}")
print(f"Velocity of A: {v_A}")
print(f"Position of B: {r_B}")
print(f"Velocity of B: {v_B}")
print(f"Position of centauri: {r_C}")
print(f"Velocity of centauri: {v_C}")

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

# Time span for one full planetary orbit
centauri_orbital_period = 2 * np.pi * np.sqrt(a_C**3 / (G * (m_A + m_B)))
t_span_realistic = (0, centauri_orbital_period)
t_eval_realistic = np.linspace(t_span_realistic[0], t_span_realistic[1], 2000)

# Initial state vector
y0_realistic = np.concatenate([r_A, r_B, r_C, v_A, v_B, v_C])

# Solve ODEs
# solution_realistic = solve_ivp(eih_accelerations, t_span_realistic, y0_realistic, t_eval=t_eval_realistic, method='RK45', args=(masses, G, c))
solution_realistic = solve_ivp(newton_accelerations, t_span_realistic, y0_realistic, t_eval=t_eval_realistic, method='RK45', args=(masses,))
# print("type(solution_realistic)", type(solution_realistic))

# Extract positions
r_A_sol_realistic, r_B_sol_realistic, r_planet_sol_realistic = (
    solution_realistic.y[:3],
    solution_realistic.y[3:6],
    solution_realistic.y[6:9],
)
# print("r_A_sol_realistic", r_A_sol_realistic)
# print("r_B_sol_realistic", r_B_sol_realistic)
# print("r_planet_sol_realistic", r_planet_sol_realistic)

# Prepare the animation
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-2 * a_C, 2 * a_C)
ax.set_ylim(-2 * a_C, 2 * a_C)
ax.set_title("Realistic Alpha Centauri System Simulation")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.grid()

# Create plot elements
star_A_plot, = ax.plot([], [], 'o', color='orange', label="Alpha Centauri A")
star_B_plot, = ax.plot([], [], 'o', color='red', label="Alpha Centauri B")
planet_plot, = ax.plot([], [], 'o', color='blue', label="Hypothetical Planet")
orbit_A_plot, = ax.plot([], [], '-', color='orange', alpha=0.5)
orbit_B_plot, = ax.plot([], [], '-', color='red', alpha=0.5)
orbit_planet_plot, = ax.plot([], [], '-', color='blue', alpha=0.5)
ax.legend()

# Initialize data for orbits
orbit_A_x, orbit_A_y = [], []
orbit_B_x, orbit_B_y = [], []
orbit_planet_x, orbit_planet_y = [], []

def init():
    star_A_plot.set_data([], [])
    star_B_plot.set_data([], [])
    planet_plot.set_data([], [])
    orbit_A_plot.set_data([], [])
    orbit_B_plot.set_data([], [])
    orbit_planet_plot.set_data([], [])
    return star_A_plot, star_B_plot, planet_plot, orbit_A_plot, orbit_B_plot, orbit_planet_plot

def update(frame):
    # Current positions of stars and planet
    r_A_x, r_A_y = r_A_sol_realistic[0][frame], r_A_sol_realistic[1][frame]
    r_B_x, r_B_y = r_B_sol_realistic[0][frame], r_B_sol_realistic[1][frame]
    r_planet_x, r_planet_y = r_planet_sol_realistic[0][frame], r_planet_sol_realistic[1][frame]
    
    # Update star and planet positions
    star_A_plot.set_data([r_A_x], [r_A_y])
    star_B_plot.set_data([r_B_x], [r_B_y])
    planet_plot.set_data([r_planet_x], [r_planet_y])
    
    # Append to orbit trails
    orbit_A_x.append(r_A_x)
    orbit_A_y.append(r_A_y)
    orbit_B_x.append(r_B_x)
    orbit_B_y.append(r_B_y)
    orbit_planet_x.append(r_planet_x)
    orbit_planet_y.append(r_planet_y)
    
    # Update orbit trails
    orbit_A_plot.set_data(orbit_A_x, orbit_A_y)
    orbit_B_plot.set_data(orbit_B_x, orbit_B_y)
    orbit_planet_plot.set_data(orbit_planet_x, orbit_planet_y)
    
    return star_A_plot, star_B_plot, planet_plot, orbit_A_plot, orbit_B_plot, orbit_planet_plot

# Create animation
frames_realistic = len(t_eval_realistic)
ani_realistic = animation.FuncAnimation(fig, update, frames=frames_realistic, init_func=init, blit=True, interval=20)

# Show the animation
plt.show()
