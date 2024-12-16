import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Constants
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
c = 3e8  # Speed of light (m/s)

# Masses (in kg)
m_A = 2.188e30  # Alpha Centauri A
m_B = 1.81e30   # Alpha Centauri B
m_planet = 5.97e24  # Hypothetical planet (Earth-mass)

# Semi-major axis (in meters)
a_binary = 3.502e12  # Semi-major axis of binary stars
a_planet_realistic = 1.496e11  # 1 AU in meters (distance from barycenter)

# Orbital eccentricity
e_binary = 0.5179

# Derived parameters
mu_binary = G * (m_A + m_B)  # Standard gravitational parameter for binary

# Initial positions and velocities
r_A = np.array([-a_binary * (m_B / (m_A + m_B)), 0, 0])  # Position of A
r_B = np.array([a_binary * (m_A / (m_A + m_B)), 0, 0])   # Position of B

v_A = np.array([0, np.sqrt(mu_binary * (1 + e_binary) / np.linalg.norm(r_A)), 0])  # Velocity of A
v_B = np.array([0, -np.sqrt(mu_binary * (1 + e_binary) / np.linalg.norm(r_B)), 0])  # Velocity of B

r_planet_realistic = np.array([a_planet_realistic, 0, 0])  # Planet position
v_planet_realistic = np.array([0, np.sqrt(G * (m_A + m_B) / a_planet_realistic), 0])  # Planet velocity

# Einstein-Infeld-Hoffmann (EIH) equations of motion
def eih_accelerations(t, y):
    # Unpack positions and velocities
    r_A, r_B, r_planet = y[:3], y[3:6], y[6:9]
    v_A, v_B, v_planet = y[9:12], y[12:15], y[15:18]

    # Distances
    r_AB = np.linalg.norm(r_A - r_B)
    r_Ap = np.linalg.norm(r_A - r_planet)
    r_Bp = np.linalg.norm(r_B - r_planet)

    # Newtonian accelerations
    a_A = -G * m_B * (r_A - r_B) / r_AB**3
    a_B = -G * m_A * (r_B - r_A) / r_AB**3
    a_planet = -G * m_A * (r_planet - r_A) / r_Ap**3 - G * m_B * (r_planet - r_B) / r_Bp**3

    return np.concatenate([v_A, v_B, v_planet, a_A, a_B, a_planet])

# Time span for one full planetary orbit
planet_orbital_period = 2 * np.pi * np.sqrt(a_planet_realistic**3 / (G * (m_A + m_B)))
t_span_realistic = (0, planet_orbital_period)
t_eval_realistic = np.linspace(t_span_realistic[0], t_span_realistic[1], 2000)

# Initial state vector
y0_realistic = np.concatenate([r_A, r_B, r_planet_realistic, v_A, v_B, v_planet_realistic])

# Solve ODEs
solution_realistic = solve_ivp(eih_accelerations, t_span_realistic, y0_realistic, t_eval=t_eval_realistic, method='RK45')

# Extract positions
r_A_sol_realistic, r_B_sol_realistic, r_planet_sol_realistic = (
    solution_realistic.y[:3],
    solution_realistic.y[3:6],
    solution_realistic.y[6:9],
)

# Prepare the animation
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-2 * a_planet_realistic, 2 * a_planet_realistic)
ax.set_ylim(-2 * a_planet_realistic, 2 * a_planet_realistic)
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
