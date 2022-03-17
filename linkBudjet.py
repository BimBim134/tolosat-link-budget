## if you don't have all the library, just pip install what you need :
# pip install TLE-tools
# pip install astropy
# pip install poliastro
# pip install tqdm

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
# download some fresh TLE : http://celestrak.com/NORAD/elements/

iridium_TLE = TLE.load('iridiumNEXT.txt') # load TLE

iridium_orb = [] # initialization
for i in range(len(iridium_TLE)): # convertion
    iridium_orb.append(iridium_TLE[i].to_orbit())

    
#%% synchronize every iridium satellite on given time t

def updateIridium(t):
    # propagate the position at given time t
    for i in range(len(iridium_orb)):
        iridium_orb[i] = iridium_orb[i].propagate(t)

updateIridium(now) # synchronise every iridium to the current time


#%% inter-satellite distance calculation definition
def dist_between(a, b):
    return np.sqrt((a.r[0]-b.r[0])**2 +
                   (a.r[1]-b.r[1])**2 +
                   (a.r[2]-b.r[2])**2)

    
#%% relative speed to doppler shift converter

def vel2doppler_shift(v, f):
    ## f = f₀(v + vr)/(v + vs)
    # v : wave speed
    # vr : receiver velocity (we consider it to be null)
    # vs : source velocity
    
    f0 = f
    c = 299792458 * (u.m / u.s)  # speed of light in vacuum
    f_shift = (f0 * c)/(c + v)
    
    return f_shift - f0

        
#%% distance to corresponding signal attenuation converter

def dist2att(d, f):
    # attenuation (dB) = 20 * log10(d) + 20 * log10(f) + 20 * log10(4π / c)
    dist = d.to(u.m)
    freq = f.to(u.Hz)
    c = 299792458  # speed of light in vacuum (in m/s)
    
    loss = 20 * np.log10(dist.value) + 20*np.log10(freq.value) + 20*np.log10(4*np.pi/c)
    
    return -loss * u.dB


#%% function to determine visibility

def theySeeEachOther(a, b):
    mean_point = (a.r + b.r)/2
    alt_mean_point = np.sqrt(mean_point[0]**2 + mean_point[1]**2 + mean_point[2]**2)
    
    return alt_mean_point > 6371 * u.km # earth diameter

#%% SIMULATION

step = 2 * u.minute # time between each step
nb_step = 500 # number of step
time_array = now + step * np.arange(nb_step) # list of dates 

result = np.zeros((len(time_array), len(iridium_orb), 6))
# [:,:,0] = distance (m)
# [:,:,1] = attenuation loss (dB)
# [:,:,2] = relative velocity (m/s)
# [:,:,3] = doppler shift (Hz)
# [:,:,4] = doppler shift rate (Hz/s)
# [:,:,5] = reachable (y/n)


print("calculating distance and attenuation...")
for t in tqdm(range(len(time_array))):
    updateIridium(time_array[t])
    tolosat_orb.propagate(time_array[t])
    
    for sat in range(len(iridium_orb)):
        dist = dist_between(tolosat_orb, iridium_orb[sat])
        result[t, sat, 0] = dist.to(u.m).value
        result[t, sat, 1] = dist2att(dist,  1.62125 * u.GHz).value
        
# the following derivations are done using finite difference method :
# https://www.wikiwand.com/en/Finite_difference_coefficient

dt = step.to(u.s).value
        
print("calculating relative velocities and doppler shift...")
for t in tqdm(np.arange(1,nb_step-1)):
    for sat in range(len(iridium_orb)):
        result[t, sat, 2] = 0.5*(result[t+1, sat, 0] - result[t-1, sat, 0])/dt
        result[t, sat, 3] = vel2doppler_shift(result[t, sat, 2]* (u.m / u.s) ,  1.62125 * u.GHz).to(u.Hz).value

for sat in range(len(iridium_orb)):
        result[0, sat, 2] = (result[1, sat, 0] - result[0, sat, 0])/dt
        result[0, sat, 3] = vel2doppler_shift(result[0, sat, 2]* (u.m / u.s) ,  1.62125 * u.GHz).to(u.Hz).value        

for sat in range(len(iridium_orb)):
        result[-1, sat, 2] = (result[-1, sat, 0] - result[-2, sat, 0])/dt
        result[-1, sat, 3] = vel2doppler_shift(result[-1, sat, 2]* (u.m / u.s) ,  1.62125 * u.GHz).to(u.Hz).value

print("calculating relative velocities and doppler shift...")
for t in np.arange(1,nb_step-1):
    for sat in range(len(iridium_orb)):
        result[t, sat, 4] = 0.5*(result[t+1, sat, 3] - result[t-1, sat, 3])/dt
        
for sat in range(len(iridium_orb)):
        result[0, sat, 4] = (result[1, sat, 3] - result[0, sat, 3])/dt

for sat in range(len(iridium_orb)):
        result[-1, sat, 4] = (result[-1, sat, 3] - result[-2, sat, 3])/dt
    
print("determine the reachability...")
for t in tqdm(range(len(time_array))):
    for sat in range(len(iridium_orb)):
        result[t, sat, 5] = int((np.abs(result[t, sat, 3]) < 37500) &
                                (np.abs(result[t, sat, 4]) < 345) &
                                theySeeEachOther(tolosat_orb, iridium_orb[sat]))
        
nb_sat_reachable = np.zeros(nb_step)
for t in range(len(time_array)):
    nb_sat_reachable[t] = np.sum(result[t,:,5])

#%% PLOT

def plot_Sat_stat(sat):
    
    plt.subplot(411)
    plt.title(iridium_TLE[sat].name)
    plt.plot(np.arange(nb_step)*dt/3600, result[:,sat,0])
    plt.ylabel('distance (m)')
    plt.grid()
    
    plt.subplot(412)
    plt.plot(np.arange(nb_step)*dt/3600, result[:,sat,1])
    plt.ylabel('attenuation loss (dB)')
    plt.grid()
    
    plt.subplot(413)
    plt.plot(np.arange(nb_step)*dt/3600, result[:,sat,3])
    plt.ylabel('doppler shift (Hz)')
    plt.grid()
    
    plt.subplot(414)
    plt.plot(np.arange(nb_step)*dt/3600, result[:,sat,4])
    plt.ylabel('doppler shift rate (Hz/s)')
    plt.xlabel('time (h)')
    plt.grid()

plt.figure()
for i in range(len(iridium_orb)):
    plt.clf()
    plot_Sat_stat(i)
    plt.pause(0.05)

#%%
plt.figure()
plt.plot(np.arange(nb_step)*dt/3600, nb_sat_reachable)
plt.ylabel('number of satellite reachable')
plt.xlabel('time (h)')
plt.grid()
