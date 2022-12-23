import starry
import matplotlib.pyplot as plt
import numpy as np

# the fake map vector
y = [1.00,  0.22,  0.19,  0.11,  0.11,  0.07,  -0.11, 0.00,  -0.05,
     0.12,  0.16,  -0.05, 0.06,  0.12,  0.05,  -0.10, 0.04,  -0.02,
     0.01,  0.10,  0.08,  0.15,  0.13,  -0.11, -0.07, -0.14, 0.06,
     -0.19, -0.02, 0.07,  -0.02, 0.07,  -0.01, -0.07, 0.04,  0.00]

starry.config.lazy = False

map = starry.Map(ydeg=5)

# computing the intensity at some latitute and longitude
print("the intesity at lat=0 and lon=0 is %f" % map.intensity(lat=0, lon=0))

# get the flux
print("the flux is %f" % map.flux())

# set the coefficient of the spherical harmonic 
map[5, -3] = -2
map[5, 0] = 2
map[5, 4] = 1

# theta = np.linspace(0, 360, 50)
# map.show(theta=theta)

# get an equirectangular (latitude-longitude) global view of the map
# map.show(projection="rect")

# load earth map
map = starry.Map(ydeg=20)
map.load("earth", sigma=0.08)

# change the orientation
map.obl = 23.5
map.inc = 60.0
# map.show()

# computing the intensity
lon = np.linspace(-180, 180, 1000)
I = map.intensity(lat=0, lon=lon)
# fig = plt.figure(figsize=(12, 5))
# plt.plot(lon, I)
# plt.xlabel("Longitude [degrees]")
# plt.ylabel("Intensity");
# plt.show()

# Computing the flux: phase curves
theta = np.linspace(0, 360, 1000)
# plt.figure(figsize=(12, 5))
# plt.plot(theta, map.flux(theta=theta))
# plt.xlabel("Rotational phase [degrees]", fontsize=20)
# plt.ylabel("Flux [normalized]", fontsize=20);
# plt.show()

# Changing the orientation of the map will change the phase curve we compute.
# plt.figure(figsize=(12, 5))
# for inc in [30, 45, 60, 75, 90]:
#     map.inc = inc
#     plt.plot(theta, map.flux(theta=theta), label="%2d deg" % inc)
# plt.legend(fontsize=10)
# plt.xlabel("Rotational phase [degrees]", fontsize=20)
# plt.ylabel("Flux [normalized]", fontsize=20);
# plt.show()

# changing the obliquity does not affect the phase curve:
# plt.figure(figsize=(12, 5))
# for obl in [30, 45, 60, 75, 90]:
#     map.obl = obl
#     plt.plot(theta, map.flux(theta=theta), label="%2d deg" % obl)
# plt.legend(fontsize=10)
# plt.xlabel("Rotational phase [degrees]", fontsize=20)
# plt.ylabel("Flux [normalized]", fontsize=20);
# plt.show()

print(map.flux.__doc__)

# Set the occultor trajectory
# moon occulting the earth
npts = 1000
time = np.linspace(0, 1, npts)
xo = np.linspace(-2.0, 2.0, npts)
yo = np.linspace(-0.3, 0.3, npts)
zo = 1.0
ro = 0.272

# Load the map of the Earth
map = starry.Map(ydeg=20)
map.load("earth", sigma=0.08)

# Compute and plot the light curve
plt.figure(figsize=(12, 5))
flux_moon = map.flux(xo=xo, yo=yo, ro=ro, zo=zo)
plt.plot(time, flux_moon)
plt.xlabel("Time [arbitrary]", fontsize=20)
plt.ylabel("Flux [normalized]", fontsize=20);
plt.show()




# fig, ax = plt.subplots(1, figsize=(5, 5))
# ax.set_xlim(-2, 2)
# ax.set_ylim(-2, 2)
# ax.axis("off")
# ax.imshow(map.render(), origin="lower", cmap="plasma", extent=(-1, 1, -1, 1))
# for n in list(range(0, npts, npts // 10)) + [npts - 1]:
#     circ = plt.Circle(
#         (xo[n], yo[n]), radius=ro, color="k", fill=True, clip_on=False, alpha=0.5
#     )
#     ax.add_patch(circ)
# plt.show()

# limb-darkening
map = starry.Map(udeg=2)
map[1] = 0.5
map[2] = 0.25

# Set the occultor trajectory
npts = 1000
time = np.linspace(0, 1, npts)
xo = np.linspace(-2.0, 2.0, npts)
yo = np.linspace(-0.3, 0.3, npts)
zo = 1.0
ro = 0.272

# Compute and plot the light curve
# plt.figure(figsize=(12, 5))
# plt.plot(time, map.flux(xo=xo, yo=yo, ro=ro, zo=zo))
# plt.xlabel("Time [arbitrary]", fontsize=20)
# plt.ylabel("Flux [normalized]", fontsize=20);
# plt.show()

map = starry.Map(ydeg=20, udeg=2)
map.load("earth", sigma=0.08)
map[1] = 0.5
map[2] = 0.25
# map.show()

# Set the occultor trajectory
npts = 1000
time = np.linspace(0, 1, npts)
xo = np.linspace(-2.0, 2.0, npts)
yo = np.linspace(-0.3, 0.3, npts)
zo = 1.0
ro = 0.272

# Set the map inclination and obliquity
map.inc = 90
map.obl = 0

# Compute and plot the light curve
# plt.figure(figsize=(12, 5))
# plt.plot(time, flux_moon, label="Limb darkening off")
# plt.plot(time, map.flux(xo=xo, yo=yo, ro=ro, zo=zo), label="Limb darkening on")
# plt.xlabel("Time [arbitrary]", fontsize=20)
# plt.ylabel("Flux [normalized]", fontsize=20)
# plt.legend();
# plt.show()

