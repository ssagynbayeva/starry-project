import numpy as np
import matplotlib.pyplot as plt
import starry

starry.config.lazy = False
starry.config.quiet = True

star = starry.Primary(starry.Map(ydeg=0, udeg=2, amp=1.0), m=0.809, r=1,theta0=106)

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

star.map[1] = 0.646 # limb-darkening coefficient 
star.map[2] = 0.048

star.map.show()

np.random.seed(123)
planet.map[1:, :] = 0.01 * np.random.randn(planet.map.Ny - 1)

planet.map.show()

system = starry.System(star, planet)
time = np.linspace(-0.25, 3.25, 10000)
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
plt.show()

system.show(t=np.linspace(0, 1, 50), window_pad=4, figsize=(8, 8), file='HAT-P-11-ani.mp4')
