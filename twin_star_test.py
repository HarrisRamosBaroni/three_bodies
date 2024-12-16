import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
e_binary = 0.2         # Binary eccentricity (updated to ensure stability)

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

# Solve the system of ODEs
t_span = (0, 5 * year)  # Simulate for 5 years
t_eval = np.linspace(*t_span, 3000)  # Time steps
y0 = initial_conditions()
solution = solve_ivp(equations_of_motion, t_span, y0, t_eval=t_eval, method='RK45')

# Extract results
r_A_sol = solution.y[:3]
r_B_sol = solution.y[3:6]
r_planet_sol = solution.y[6:9]

# Visualization
fig, ax = plt.subplots(figsize=(8, 8))

# Axis limits
axis_limit = 1.5 * a_planet
ax.set_xlim(-axis_limit, axis_limit)
ax.set_ylim(-axis_limit, axis_limit)

ax.set_title("Binary Stars and a Circumbinary Planet")
ax.set_xlabel("X Position (m)")
ax.set_ylabel("Y Position (m)")

# Plot objects
star_A_plot, = ax.plot([], [], 'o', color='orange', label="Star A", markersize=10)
star_B_plot, = ax.plot([], [], 'o', color='yellow', label="Star B", markersize=10)
planet_plot, = ax.plot([], [], 'o', color='blue', label="Planet", markersize=5)
planet_orbit, = ax.plot([], [], '-', alpha=0.5, color='blue')

# Initialization
def init():
    star_A_plot.set_data([], [])
    star_B_plot.set_data([], [])
    planet_plot.set_data([], [])
    planet_orbit.set_data([], [])
    return star_A_plot, star_B_plot, planet_plot, planet_orbit

# Update function
def update(frame):
    # Update star positions
    star_A_plot.set_data([r_A_sol[0, frame]], [r_A_sol[1, frame]])
    star_B_plot.set_data([r_B_sol[0, frame]], [r_B_sol[1, frame]])

    # Update planet position and orbit
    planet_plot.set_data([r_planet_sol[0, frame]], [r_planet_sol[1, frame]])
    planet_orbit.set_data(r_planet_sol[0, :frame+1], r_planet_sol[1, :frame+1])

    return star_A_plot, star_B_plot, planet_plot, planet_orbit

# Animation
ani = animation.FuncAnimation(fig, update, frames=len(t_eval), init_func=init, blit=True, interval=30)
ax.legend()
plt.show()
