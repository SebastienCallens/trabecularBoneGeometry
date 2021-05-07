# -*- coding: utf-8 -*-
"""
Calculation of Radial Distribution Function

Developed by Sebastien Callens
"""
import numpy as np
import matplotlib.pyplot as plt
import igl
from scipy import spatial
import os
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

# %% Create RDF data
# Load in data
directory = 'outputFolder' 
filename = 'test_r5'
k1 = np.load(os.path.join(directory,filename)+'_k1.npy')
k2 = np.load(os.path.join(directory,filename)+'_k2.npy')
f = np.load(os.path.join(directory,filename)+'_f.npy')
v = np.load(os.path.join(directory,filename)+'_v.npy')
faceArea = igl.doublearea(v,f)/2
H = H = 0.5*(k1+k2)
H = np.nan_to_num(H)
faceH = (H[f[:,0]]+H[f[:,1]]+H[f[:,2]])/3
avgEdgeLength = igl.avg_edge_length(v,f)

# Select a characteristic length, in this case (1/Sv)=volume/area. 
# So we define the characteristic length as:
charLength = (4**3)/np.sum(faceArea)

rTilde = 0.1
r = rTilde*charLength
# Specify spherical shell region 
rDelta = (1/10)*charLength
rmax = r+rDelta/2
rmin = r-rDelta/2

## Process the H range that we will consider. We will use the 99.7% confidence interval: Hmean -/+ 3 std
# Calculate mean and std
faceHmean = np.mean(faceH)
faceHstd = np.std(faceH)
nbins = 100
perc05 = np.percentile(faceH,0.5)
perc995 = np.percentile(faceH,99.5)
faceHbins = np.linspace(perc05,perc995,nbins+1)
# Digitize faceH. We will ignore the first and the last bin because they have a very high bin count (because everything
# outside of the range is included in those bins)
faceHdigit = np.digitize(faceH,faceHbins)

# Create the area histogram based on the total distribution of mean curvature, normalized by total area
A_vH2 = np.histogram(faceH,bins=nbins,range=(perc05,perc995),weights=faceArea)[0]
A_vH2 /= np.sum(faceArea)

# Calculate the barycenters (centroids) of the triangles
baryCenters = igl.barycenter(v,f)
# Create a tree structure (this is a little time-consuming)
tree = spatial.KDTree(baryCenters)

# We will loop over every bin of H1, and sample it with nSampleH1 points (e.g. 100 points).
# This means that we take 100 random points for every H1, and find the distant points for every one of those sample
# points, and compute the H2 histogram for the distant points
nSampleH1 = 1000
pointsIdx = np.arange(len(baryCenters)) # just an increasing vector with idx of the points in the system

# Preallocate
G = np.zeros((nbins,nbins))
Gsavgol = np.zeros((nbins,nbins))
Gnorm = np.zeros((nbins,nbins))
# %%
for i in tqdm(np.arange(nbins)+1):
    # we go bin by bin over the H1 range. In every bin, we select 100 points at random (100 points in the structure that
    # have this value of H). We go from i=1 (!) until i=nbins. By doing this, the first bin (0) and the last bin (201) of 
    # faceHdigit is excluded (desired)
    # Check where faceHdigit is equal to i
    boolCurrH1 = faceHdigit == i
    pointsCurrH1 = pointsIdx[boolCurrH1] # get the indices of all the points belonging to the current bin
    # Now we want to sample the points in the current bin. So we want nSampleH1 random points (or a bit less, since we 
    # only take the unique points)
    if len(pointsCurrH1)<nSampleH1:
        samplePoints = np.copy(pointsCurrH1) # if there are not enough points in the bin, we use all the points in the bin
    else:
        samplePoints = np.unique(np.random.choice(pointsCurrH1,nSampleH1))
    # Preallocate
    H2hist = np.zeros((len(samplePoints),nbins))
    Ahist = np.zeros((len(samplePoints),nbins))
    for j in np.arange(len(samplePoints)):
        currentPoint = samplePoints[j]
        # Extract points that are r away, by subtracting ball2 from ball1
        ball1 = tree.query_ball_point(baryCenters[currentPoint,:],rmax)
        ball2 = tree.query_ball_point(baryCenters[currentPoint,:],rmin)
        distPoints = list(set(ball1)-set(ball2))
        # Find H2 values of the distant points
        H2dist = faceH[distPoints]
        # Find the areas of the distant points (patches)
        Adist = faceArea[distPoints]
        # Calculate histogram of H2 (bincount) for all points
        H2hist[j,:] = np.histogram(H2dist,bins=nbins,range=(perc05,perc995))[0]
        # Calculate histogram of areas based on histogram of H2 (so we sum all the areas that correspond to every 
        # bin of H2hist). This is possible by weighing with Adist
        Ahist[j,:] = np.histogram(H2dist,bins=nbins,range=(perc05,perc995),weights=Adist)[0]
        # normalize by total area of distant points
        Ahist[j,:] /= np.sum(Adist)
    # Now we average all the H2-area histograms for this particular H1 bin
    G[:,i-1] = np.sum(Ahist,axis=0)/len(samplePoints)
    # Now we normalize by the "uniform" area distribution A_vH2
    Gnorm[:,i-1] = np.divide(G[:,i-1],A_vH2)
Gnorm2 = gaussian_filter(Gnorm,sigma=2)
np.save('Gnorm2.npy',Gnorm2)
# %% Plot data

faceHrange = np.linspace(perc05,perc995,nbins)
Xh,Yh = np.meshgrid(faceHrange/(np.sum(faceArea)/(4**3)),faceHrange/(np.sum(faceArea)/(4**3)))
cPlot = plt.contourf(Xh,Yh,Gnorm2,levels = np.linspace(0.75,1.5,25),cmap ='coolwarm',extend='both')
for a in cPlot.collections:
    a.set_edgecolor('face')
xmin = np.min(faceHrange)/(np.sum(faceArea)/(4**3))
xmax = np.max(faceHrange)/(np.sum(faceArea)/(4**3))
ymin = np.min(faceHrange)/(np.sum(faceArea)/(4**3))
ymax = np.max(faceHrange)/(np.sum(faceArea)/(4**3))
plt.gca().set_aspect("equal")
plt.hlines(0,xmin,xmax,colors='w',linestyles='solid')
plt.vlines(0,ymin,ymax,colors='w',linestyles='solid')
cbar = plt.colorbar(ticks=[0.75,1.0,1.25,1.5])
plt.xlabel(r'$H_1/S_v$')
plt.ylabel(r'$H_2/S_v$')
titleStr = 'RDF - ICF05-'+r'$\tilde{r}$='+str(rTilde)+'Sv - '+str(nSampleH1)
saveStr = 'RDF-ICF05-'+'r='+str(rTilde)+'Sv-'+str(nSampleH1)
plt.title(titleStr)
plt.savefig(saveStr+'.svg',format='svg',dpi=600)
plt.show()
