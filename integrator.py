import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x*(1-x)

def rk4(f, x0, dt):
    k1 = f(x0)*dt
    k2 = f(x0+(1/2)*k1)*dt
    k3 = f(x0+(1/2)*k2)*dt
    k4 = f(x0+k3)*dt
    x_1 = x0 + (1/6)*(k1+2*k2+2*k3+k4)
    return x_1

# single initial value
x0 = 0.5
dt = 0.01

t = np.arange(0, 10, dt)
x = np.zeros_like(t)

x[0] = x0

for k in range(t.shape[0]-1):
    x_k = x[k]
    x[k+1] = rk4(f, x_k, dt)


#multiple initial values
dt = 0.5
initial_values = [0.15, 0.01]
t = np.arange(0, 10, dt)
x = np.zeros_like(t)

for x0 in initial_values:
    x[0] = x0
    for k in range(t.shape[0]-1):
        x_k = x[k]
        x[k+1] = rk4(f, x_k, dt)

    plt.plot(t, x)

plt.grid()
plt.xlim((0, 10))
plt.ylim((0, 2))
plt.savefig('plots/plot.png')