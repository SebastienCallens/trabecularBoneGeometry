
# -*- coding: utf-8 -*-
"""
Compute Interface Shape Distributions (ISD)
Developed by Sebastien Callens
"""

import numpy as np
import matplotlib.pyplot as plt
import igl
import os

# %% Load data
# Load data from "meshAnalyze.py"
directory = 'outputFolder' 
filename = 'test_r5'
k1 = np.load(os.path.join(directory,filename)+'_k1.npy')
k2 = np.load(os.path.join(directory,filename)+'_k2.npy')
f = np.load(os.path.join(directory,filename)+'_f.npy')
v = np.load(os.path.join(directory,filename)+'_v.npy')
faceArea = igl.doublearea(v,f)/2
# Clip range for visualization
lowk1 = np.mean(k1)-10*np.std(k1)
highk1 = np.mean(k1)+10*np.std(k2)
k1clipped = np.clip(k1,lowk1,highk1)
lowk2 = np.mean(k2)-10*np.std(k2)
highk2= np.mean(k2)+10*np.std(k2)
k2clipped = np.clip(k2,lowk2,highk2)

k1_all = k1clipped
k2_all = k2clipped
faceArea_all = faceArea
k1_all = np.concatenate((k1_all,[lowk1,highk1]))
k2_all = np.concatenate((k2_all,[lowk2,highk2]))
faceArea_all = np.concatenate((faceArea_all,faceArea_all[0:2]))

# %% Create k1k2 ISD
# Normalize data using a length measure (S/V). First calculate volume of bounding box (sampleVol)
sampleVol = (np.max(v[:,0])-np.min(v[:,0]))*(np.max(v[:,1])-np.min(v[:,1]))*(np.max(v[:,2])-np.min(v[:,2]))
normk1 = k1_all/(np.sum(faceArea_all)/sampleVol)
normk2 = k2_all/(np.sum(faceArea_all)/sampleVol)
# Create histogram (but don't plot it)
fig, ax = plt.subplots()
h = plt.hist2d(normk1,normk2,bins=500,weights=faceArea_all)
plt.clf()
# Now we normalize the histogram data (h[0])
hnew = h[0]/(np.sum(h[0])*(h[1][1]-h[1][0])*(h[2][1]-h[2][0]))
# Plot using contour
xmin = -4
xmax = 1
ymin = -1
ymax = 4
xregion = np.arange(-2,2,0.01)
yregion = xregion
xh,yh = np.meshgrid(h[1][0:-1],h[2][0:-1])
cPlot = plt.contourf(xh,yh,hnew.T,cmap='coolwarm',levels=25)
for a in cPlot.collections:
    a.set_edgecolor('face')
plt.xlim(right=xmax)
plt.xlim(left=xmin)
plt.ylim(top=ymax)
plt.ylim(bottom=ymin)
plt.hlines(0,np.min(normk1),np.max(normk1),colors='w',linestyles='dotted')
plt.vlines(0,np.min(normk2),np.max(normk2),colors='w',linestyles='dotted')
plt.plot([-2,2],[-2,2],'w:')
plt.plot([-8,2],[8,-2],'w:')
plt.fill_between(xregion,yregion,-2,facecolor='white')
cbar = plt.colorbar()
cbar.set_label('PDF', rotation=270, labelpad=20, fontsize=12)
plt.xlabel(r'$\kappa_1/S_v$ [-]',fontsize=14)
plt.ylabel(r'$\kappa_2/S_v$ [-]',fontsize=14)
titleStr = 'ISD - '+filename
plt.title(titleStr,fontsize=12)
ax.set_aspect('equal')
#plt.savefig(titleStr+'.svg',format='svg',dpi=600)
plt.show()

# %% Create K vs H ISD
normK = k1_all*k2_all*(sampleVol/np.sum(faceArea_all))**2
normH = 0.5*(k1_all+k2_all)*(sampleVol/np.sum(faceArea_all))
fig, ax = plt.subplots()
h2 = plt.hist2d(normH,normK,bins=100,weights=faceArea_all,range=[[-5,5],[-10,10]])
plt.clf()
h2new = h2[0]/(np.sum(h2[0])*(h2[1][1]-h2[1][0])*(h2[2][1]-h2[2][0]))
xmin = -5
xmax = 5
ymin = -10
ymax = 10
xh,yh = np.meshgrid(h2[1][0:-1],h2[2][0:-1])
plt.hlines(0,xmin,xmax,colors='w',linestyles='dotted')
plt.vlines(0,ymin,ymax,colors='w',linestyles='dotted')
cPlot = plt.contourf(xh,yh,h2new.T,cmap='coolwarm',levels=25)
for a in cPlot.collections:
    a.set_edgecolor('face')

xcurve = np.arange(xmin,xmax,0.005)
ycurve = xcurve**2
plt.fill_between(xcurve,ycurve,ymax,facecolor='white')
plt.xlim(right=xmax)
plt.xlim(left=xmin)
plt.ylim(top=ymax)
plt.ylim(bottom=ymin)
cbar = plt.colorbar()
cbar.set_label('PDF', rotation=270, labelpad=20, fontsize=12)
plt.xlabel(r'$H/S_v$ [-]',fontsize=14)
plt.ylabel(r'$K/S_v^2$ [-]',fontsize=14)
titleStr = 'ISD_KH - '+filename
plt.title(titleStr,fontsize=12)
#plt.savefig(titleStr+'.svg',format='svg',dpi=600)
ax.set_aspect('equal')
plt.show()


# %% Create D (net curvature) vs H plot
normD = np.sqrt((k1_all**2+k2_all**2)/2)*(sampleVol/np.sum(faceArea_all))
fig, ax = plt.subplots()
h3 = plt.hist2d(normH,normD,bins=500,weights=faceArea_all,cmap='coolwarm')
plt.clf()
h3new = h3[0]/(np.sum(h3[0])*(h3[1][1]-h3[1][0])*(h3[2][1]-h3[2][0]))
xmin = -2
xmax = 2
ymin = 0
ymax = 4
xh,yh = np.meshgrid(h3[1][0:-1],h3[2][0:-1])
cPlot = plt.contourf(xh,yh,h3new.T,cmap='coolwarm',levels=25)
cbar = plt.colorbar()
for a in cPlot.collections:
    a.set_edgecolor('face')
plt.vlines(0,ymin,ymax,colors='w',linestyles='dotted')
xcurve1 = np.arange(0,xmax,0.005)
ycurve1 = xcurve1
xcurve2 = np.arange(xmin,0,0.005)
ycurve2 = -xcurve2
plt.fill_between(xcurve1,ycurve1-0.001,0,facecolor='white')
plt.fill_between(xcurve2,ycurve2-0.001,0,facecolor='white')
plt.xlim(right=xmax)
plt.xlim(left=xmin)
plt.ylim(top=ymax)
plt.ylim(bottom=ymin)
cbar.set_label('PDF', rotation=270, labelpad=20, fontsize=12)
plt.xlabel(r'$H/S_v$ [-]',fontsize=14)
plt.ylabel(r'$D/S_v$ [-]',fontsize=14)
titleStr = 'ISD_DH - '+filename
plt.title(titleStr,fontsize=12)
#plt.savefig(titleStr+'.svg',format='svg',dpi=600)
ax.set_aspect('equal')
plt.show()
