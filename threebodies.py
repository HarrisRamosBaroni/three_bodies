import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Gravitational constant
G = 6.67430e-11  # m^3 kg^-1 s^-2

# Masses of the three bodies (in kg)
m1, m2, m3 = 1e24, 1e24, 1e24

# Initial positions and velocities (in meters, meters per second)
r1, r2, r3 = np.array([1e11, 0]), np.array([0, 1e11]), np.array([1e11, 1e11])
v1, v2, v3 = np.array([0, 1e3]), np.array([0, -1e3]), np.array([1e3, -1e3])

# Function to compute gravitational force
def grav_force(r1, r2, m1, m2):
    r = np.linalg.norm(r2 - r1)
    return G * m1 * m2 * (r2 - r1) / r**3

# Equations of motion
def equations(t, y):
    r1 = np.array([y[0], y[1]])
    r2 = np.array([y[2], y[3]])
    r3 = np.array([y[4], y[5]])
    v1 = np.array([y[6], y[7]])
    v2 = np.array([y[8], y[9]])
    v3 = np.array([y[10], y[11]])

    # Gravitational forces
    F12 = grav_force(r1, r2, m1, m2)
    F13 = grav_force(r1, r3, m1, m3)
    F23 = grav_force(r2, r3, m2, m3)

    # Accelerations
    a1 = (F12 + F13) / m1
    a2 = (-F12 + F23) / m2
    a3 = (-F13 - F23) / m3

    # Return derivatives: position and velocity
    dydt = [v1[0], v1[1], v2[0], v2[1], v3[0], v3[1], a1[0], a1[1], a2[0], a2[1], a3[0], a3[1]]
    return dydt

# Initial conditions
y0 = [*r1, *r2, *r3, *v1, *v2, *v3]

# Time span
t_span = (0, 1e5)  # seconds
t_eval = np.linspace(0, 1e5, 1000)

# Solve ODEs
solution = solve_ivp(equations, t_span, y0, t_eval=t_eval) # solution has t and y: t is the time points, y is the values of the solution at each time point
print("Solution:", solution)
print("solution.y.shape:", solution.y.shape) # (12, 1000)
print("solution.y[0].shape:", solution.y[0].shape)
print("solution.t.shape:", solution.t.shape)

# Plot the orbits
# plt.plot(solution.t, solution.y[0], label="Body 1")
# plt.plot(solution.t, solution.y[1], label="Body 2")
# plt.plot(solution.t, solution.y[2], label="Body 3")
# plt.legend()
# plt.show()

# Plot the orbits
plt.plot(solution.y[0], solution.y[1], label="Body 1")
plt.legend()
plt.show()

plt.plot(solution.y[2], solution.y[3], label="Body 2")
plt.legend()
plt.show()

plt.plot(solution.y[4], solution.y[5], label="Body 3")
plt.legend()
plt.show()


# Plot the orbits in 3D