#% general library import
import numpy as np
import matplotlib.pyplot as plt

# TLE tools import
from tletools import TLE

# poly astro library import
from astropy import units as u
import poliastro as pa

# to plot at current time
from astropy.time import Time

# just a progress bar
from tqdm import tqdm

#%% variable declaration
now = Time.now() # get the actual time

tolosat_orb = pa.twobody.orbit.Orbit.from_classical(
    pa.bodies.Earth,    # main attractor
    6878 * u.km,        # Semi-major axis
    0.002 * u.one,      # Eccentricity
    97.4 * u.deg,       # Inclination
    0.187 * u.rad,      # RAAN (Right ascension of the ascending node)
    np.pi/2 * u.rad,    # Argument of the pericenter
    0 * u.deg,          # true anomaly
    now                 # epoch (time)
)

#%% TLE --> orbit conversion
# this cell convert list of TLE to list of poliastro.orbit

iridium_TLE = TLE.load('iridiumTLE.txt') # load TLE

iridium_orb = [] # initialization
for i in range(len(iridium_TLE)): # convertion
    iridium_orb.append(iridium_TLE[i].to_orbit())
    
#%% synchronize every iridium satellite on given time t

def updateIridium(t):
    # propagate the position at given time t
    for i in range(len(iridium_orb)):
        iridium_orb[i] = iridium_orb[i].propagate(t)

updateIridium(now)

#%% distance calculation definition
def dist_between(a, b):
    return np.sqrt((a.r[0]-b.r[0])**2 +
                   (a.r[1]-b.r[1])**2 +
                   (a.r[2]-b.r[2])**2)

#%% minimum distance from constellation definition

min_dist = []
min_dist_name = []

def minDist():
    distFromConst = []
    distFromConst_name = []
    
    for i in range(len(iridium_orb)):
        distFromConst.append(dist_between(tolosat_orb,iridium_orb[i]))
        distFromConst_name.append(iridium_TLE[i].name)

    index_of_min = distFromConst.index(min(distFromConst))
    
    min_dist.append(distFromConst[index_of_min])
    min_dist_name.append(distFromConst_name[index_of_min])

#%% sweep in time

step = 10 * u.s
nb_step = 1000

time_array = 10 * np.arange(nb_step)

for i in tqdm(range(nb_step)):
    minDist()
    updateIridium(step)
    tolosat_orb.propagate(step)        

for i in range(len(min_dist)):
    min_dist[i] = min_dist[i].value 

min_dist = np.array(min_dist)

min_dist_mean = np.mean(min_dist)*np.ones(min_dist.shape)

#%%

plt.figure()

plt.plot(time_array,min_dist)
plt.plot(time_array, min_dist_mean, linestyle='--')

plt.grid()
plt.xlabel('time (s)')
plt.ylabel('distance (km)')
plt.title('minimal distance between tolosat and iridium constellation')

plt.legend(['minimal distance','mean of minimal distance'])

