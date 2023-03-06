import numpy as np
import matplotlib.pyplot as plt
import starry
import matplotlib.animation as animation

starry.config.lazy = False
starry.config.quiet = True

map = starry.Map(ydeg=15, udeg=2,amp=1)
star = starry.Primary(map, m=0.809, r=1,theta0=106)

planet = starry.kepler.Secondary(
    starry.Map(ydeg=5, amp=5e-3),  # the surface map
    m=7.0257708e-5,  # mass in solar masses
    r=0.3,  # radius in solar radii
    inc=88.99, # orbital inclination
    porb=1.,  # orbital period in days
    w=-162.149,  # Argument of periastron (little omega)
    Omega=106,
    ecc=0.265,  # eccentricity
    # t0=0.098,  # time of transit in days
)

map[1] = 0.646 # limb-darkening coefficient 
map[2] = 0.048

star.map.show()

np.random.seed(123)
planet.map[1:, :] = 0.01 * np.random.randn(planet.map.Ny - 1)

planet.map.show()

system = starry.System(star, planet)
time = np.linspace(-0.25, 3.25, 1000)
# time = np.linspace(0, 1, 50)
flux_system = system.flux(time)

starry.config.lazy = False
starry.config.quiet = True

plt.plot(time, flux_system)
plt.xlabel("time [days]")
plt.ylabel("system flux");
plt.show()


flux_star, flux_planet = system.flux(time, total=False)
plt.plot(time, flux_star)
plt.xlabel("time [days]")
plt.ylabel("stellar flux");
ax = plt.gca()
print(ax.get_xlim(), ax.get_ylim())
plt.show()

fig = plt.figure() 
ax = plt.axes(xlim=(-0.425, 3.425), ylim=(0.88, 1.0056)) 
line, = ax.plot([], [], lw=2) 

# initialization function 
def init(): 
    # creating an empty plot/frame 
    line.set_data([], []) 
    return line,

def animate(i): 
    
    line.set_data(time[:i*10], flux_star[:i*10]) 
    return line,

anim = animation.FuncAnimation(fig, animate, init_func=init,frames=100, interval=20, blit=False)
anim.save('coil.gif',writer='imagemagick') 

system.show(t=np.linspace(0, 1, 50), window_pad=4, figsize=(8, 8))
