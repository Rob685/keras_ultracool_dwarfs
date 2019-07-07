import numpy as np
import sys, os, os.path, time
from astropy.table import Table
import glob
import gzip
import matplotlib.pyplot as plt

# Start timing counter to see how fast it runs
start = time.time()

# Path where the data files are stored
path  = '/Users/roberttejada/Desktop/SM_All'

# Array of files. You can either use '*.gz' for the compressed files or '*.csv' for the uncompressed files
files = glob.glob( os.path.join(path, '*.gz') ) # use this if you have .gz files in your directory
#files = glob.glob( os.path.join(path, '*.csv') ) # use this if you have .csv files in your directory

# empty arrays for the data you want to plot
i_z  = np.empty([0])
r_i  = np.empty([0])
g_r  = np.empty([0])
g_i  = np.empty([0])
u_g  = np.empty([0])

# Start the loop to go through each file
for infile in files:

    # Get the data from each table
    t = Table.read(infile, format='csv') 

    # Example of getting colors from each table
    i_z = np.ma.concatenate( [i_z, t['i_psf'].data - t['z_psf'].data] )
    r_i = np.ma.concatenate( [r_i, t['r_psf'].data - t['i_psf'].data] )
    g_r = np.ma.concatenate( [g_r, t['g_psf'].data - t['r_psf'].data] )
    g_i = np.ma.concatenate( [g_i, t['g_psf'].data - t['i_psf'].data] )
    u_g = np.ma.concatenate( [u_g, t['u_psf'].data - t['g_psf'].data] )
    
# See how long the loop took in minutes
print((time.time() - start) / 60.)
    
# Plot the data when the loop is finished
plt.scatter(i_z, r_i, s = 5, c= 'k', alpha=0.1)
plt.xlabel(r'I - Z')
plt.ylabel(r'R - I')
plt.xlim(0, 4)
plt.ylim(0, 4)
plt.title('R - I v. I - Z: All SkyMapper')
plt.show()

plt.scatter(r_i, g_r, s = 5, c= 'k', alpha=0.1)
plt.xlabel(r'R - I')
plt.ylabel(r'G - R')
plt.xlim(0, 4)
plt.ylim(0, 4)
plt.title('G - R v. R - I: All SkyMapper')
plt.show()

plt.scatter(g_i, u_g, s = 5, c = 'k', alpha=0.1)
plt.xlabel(r'G - I')
plt.ylabel(r'U - G')
plt.xlim(0, 4)
plt.ylim(0, 4)
plt.title('U - G v. G - I: All SkyMapper')
plt.show()

# Or you can create tables for later to save yourself time
#Twrite = Table( [Xaxis, Yaxis], names=('g-i', 'u-g'))
#Twrite.write(path + 'GI_UG_ColorTable.fits', overwrite=True)
    
