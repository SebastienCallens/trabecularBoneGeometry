# -*- coding: utf-8 -*-
"""
Process single mesh: 
    - Load mesh (e.g. stl)
    - Calculate curvatures
    - Write to VTK for visualization in Paraview
    
Developed by Sebastien Callens
"""

import numpy as np
import trimesh as tr
import igl
from collections import Counter
from tqdm import tqdm
import os

# %%

# Specify input directory
directory = "inputFolder"
# Specify output directory
directorySave = "outputFolder"
filename = "test.stl"

# Load mesh
meshProc = tr.load(os.path.join(directory,filename))
meshverts = meshProc.vertices.view(np.ndarray)
meshfaces = meshProc.faces.view(np.ndarray)

# Remove the disconnected components
componentsMesh = igl.face_components(meshfaces)
# Find the most commonly occuring number (largest component)
most_common,num_most_common = Counter(componentsMesh).most_common(1)[0] 
boolComponent = componentsMesh==most_common
meshfaces = meshfaces[boolComponent]
# Make trimesh format
smoothMesh = tr.Trimesh(vertices=meshverts,faces=meshfaces)
smoothMesh.remove_unreferenced_vertices()
smoothMesh.remove_duplicate_faces()
meshverts = smoothMesh.vertices.view(np.ndarray)
meshfaces = smoothMesh.faces.view(np.ndarray)
# Adjust meshfaces 
diff1 = meshfaces[:,1]-meshfaces[:,0]
diff2 = meshfaces[:,2]-meshfaces[:,0]
diff3 = meshfaces[:,2]-meshfaces[:,1]
bool1 = diff1==0
bool2 = diff2==0
bool3 = diff3==0
boolArray = np.array([bool1,bool2,bool3])
boolRow = np.any(boolArray.T,axis=1)
boolRow = ~boolRow
meshfaces = meshfaces[boolRow]

# %% Calculate curvatures
# Specify region of interest for curvature estimation
curvR = 5
# Curvature estimation using method from Panozzo et al.
p1,p2,k1,k2 = igl.principal_curvature(meshverts,meshfaces,radius=curvR)

# Gaussian curvature
K = k1*k2
K = np.nan_to_num(K)
# Mean curvature
H = -0.5*(k1+k2)
H = np.nan_to_num(H)
# Net curvature
D = np.sqrt((k1**2+k2**2)/2)
D = np.nan_to_num(D)
# Clipping for visualization
lowK = np.mean(K)-3*np.std(K)
highK = np.mean(K)+3*np.std(K)
Kclipped = np.clip(K,lowK,highK)
lowH = np.mean(H)-3*np.std(H)
highH = np.mean(H)+3*np.std(H)
Hclipped = np.clip(H,lowH,highH)
lowD = np.mean(D)-3*np.std(D)
highD = np.mean(D)+3*np.std(D)
Dclipped = np.clip(D,lowD,highD)
# area of the faces
faceArea = igl.doublearea(meshverts,meshfaces)/2
# volume of bounding box
sampleVol = (np.max(meshverts[:,0])-np.min(meshverts[:,0]))*(np.max(meshverts[:,1])-np.min(meshverts[:,1]))*(np.max(meshverts[:,2])-np.min(meshverts[:,2]))
# average vertex curvatures onto faces  (in hindsight, not sure why this is needed)
faceK = (K[meshfaces[:,0]]+K[meshfaces[:,1]]+K[meshfaces[:,2]])/3
faceKclipped = (Kclipped[meshfaces[:,0]]+Kclipped[meshfaces[:,1]]+Kclipped[meshfaces[:,2]])/3
faceH = (H[meshfaces[:,0]]+H[meshfaces[:,1]]+H[meshfaces[:,2]])/3
faceHclipped = (Hclipped[meshfaces[:,0]]+Hclipped[meshfaces[:,1]]+Hclipped[meshfaces[:,2]])/3
facek1 = (k1[meshfaces[:,0]]+k1[meshfaces[:,1]]+k1[meshfaces[:,2]])/3
facek2 = (k2[meshfaces[:,0]]+k2[meshfaces[:,1]]+k2[meshfaces[:,2]])/3
faceD = (D[meshfaces[:,0]]+D[meshfaces[:,1]]+D[meshfaces[:,2]])/3
faceDclipped = (Dclipped[meshfaces[:,0]]+Dclipped[meshfaces[:,1]]+Dclipped[meshfaces[:,2]])/3

# normalize K, H and D (optional)
KclippedNorm = Kclipped*(sampleVol/np.sum(faceArea))**2
HclippedNorm = Hclipped*(sampleVol/np.sum(faceArea))
DclippedNorm = Dclipped*(sampleVol/np.sum(faceArea))
k1Norm = k1*(sampleVol/np.sum(faceArea))
k2Norm = k2*(sampleVol/np.sum(faceArea))

filenameShort = filename[0:-4]

# Save vertices, faces, and principal curvatures
np.save(os.path.join(directorySave, filenameShort)+'_r'+str(curvR)+'_v',meshverts)
np.save(os.path.join(directorySave, filenameShort)+'_r'+str(curvR)+'_f',meshfaces)
np.save(os.path.join(directorySave, filenameShort)+'_r'+str(curvR)+'_k1',facek1)
np.save(os.path.join(directorySave, filenameShort)+'_r'+str(curvR)+'_k2',facek2)

#%% Write vtk 
vtkStr_K = os.path.join(directorySave,filenameShort)+'_K'+'.vtk'
vtkStr_H = os.path.join(directorySave,filenameShort)+'_H'+'.vtk'
vtkStr_D = os.path.join(directorySave,filenameShort)+'_D'+'.vtk'
vtkStr_k1 = os.path.join(directorySave,filenameShort)+'_k1'+'.vtk'
vtkStr_k2 = os.path.join(directorySave,filenameShort)+'_k2'+'.vtk'

fK = open(vtkStr_K,'w')
fH = open(vtkStr_H,'w')
fD = open(vtkStr_D,'w')
fk1 = open(vtkStr_k1,'w')
fk2 = open(vtkStr_k2,'w')

fK.write('# vtk DataFile Version 3.0\n')
fH.write('# vtk DataFile Version 3.0\n')
fD.write('# vtk DataFile Version 3.0\n')
fk1.write('# vtk DataFile Version 3.0\n')
fk2.write('# vtk DataFile Version 3.0\n')

fK.write('vtk output\n')
fH.write('vtk output\n')
fD.write('vtk output\n')
fk1.write('vtk output\n')
fk2.write('vtk output\n')

fK.write('ASCII\n')
fH.write('ASCII\n')
fD.write('ASCII\n')
fk1.write('ASCII\n')
fk2.write('ASCII\n')

fK.write('DATASET POLYDATA\n')
fH.write('DATASET POLYDATA\n')
fD.write('DATASET POLYDATA\n')
fk1.write('DATASET POLYDATA\n')
fk2.write('DATASET POLYDATA\n')

fK.write('POINTS '+str(len(meshverts))+' float\n')
fH.write('POINTS '+str(len(meshverts))+' float\n')
fD.write('POINTS '+str(len(meshverts))+' float\n')
fk1.write('POINTS '+str(len(meshverts))+' float\n')
fk2.write('POINTS '+str(len(meshverts))+' float\n')
for row in tqdm(meshverts):
    fK.write(' '.join(str(item) for item in row)+'\n')
    fH.write(' '.join(str(item) for item in row)+'\n')
    fD.write(' '.join(str(item) for item in row)+'\n')
    fk1.write(' '.join(str(item) for item in row)+'\n')
    fk2.write(' '.join(str(item) for item in row)+'\n')
    
fK.write('\n')
fH.write('\n')
fD.write('\n')
fk1.write('\n')
fk2.write('\n')
fK.write('POLYGONS '+str(len(meshfaces))+' '+str(4*len(meshfaces))+'\n')  
fH.write('POLYGONS '+str(len(meshfaces))+' '+str(4*len(meshfaces))+'\n')  
fD.write('POLYGONS '+str(len(meshfaces))+' '+str(4*len(meshfaces))+'\n')   
fk1.write('POLYGONS '+str(len(meshfaces))+' '+str(4*len(meshfaces))+'\n')
fk2.write('POLYGONS '+str(len(meshfaces))+' '+str(4*len(meshfaces))+'\n')
for row2 in tqdm(meshfaces):
    fK.write(str(3)+' ')
    fH.write(str(3)+' ')
    fD.write(str(3)+' ')
    fk1.write(str(3)+' ')
    fk2.write(str(3)+' ')
    fK.write(' '.join(str(item) for item in row2)+'\n')
    fH.write(' '.join(str(item) for item in row2)+'\n')
    fD.write(' '.join(str(item) for item in row2)+'\n')
    fk1.write(' '.join(str(item) for item in row2) + '\n')
    fk2.write(' '.join(str(item) for item in row2) + '\n')
             
fK.write('\n')
fH.write('\n')
fD.write('\n')
fk1.write('\n')
fk2.write('\n')
fK.write('POINT_DATA '+str(len(meshverts))+'\n')
fH.write('POINT_DATA '+str(len(meshverts))+'\n')
fD.write('POINT_DATA '+str(len(meshverts))+'\n')
fk1.write('POINT_DATA '+str(len(meshverts))+'\n')
fk2.write('POINT_DATA '+str(len(meshverts))+'\n')
fK.write('SCALARS gaussCurv float\n')
fH.write('SCALARS meanCurv float\n')
fD.write('SCALARS netCurv float\n')
fk1.write('SCALARS prinCurv1 float\n')
fk2.write('SCALARS prinCurv2 float\n')
fK.write('LOOKUP_TABLE default\n')
fH.write('LOOKUP_TABLE default\n')
fD.write('LOOKUP_TABLE default\n')
fk1.write('LOOKUP_TABLE default\n')
fk2.write('LOOKUP_TABLE default\n')
for el in tqdm(np.arange(len(Kclipped))):
    fK.write(str(KclippedNorm[el])+'\n')
    fH.write(str(HclippedNorm[el])+'\n')
    fD.write(str(DclippedNorm[el])+'\n')
    fk1.write(str(k1Norm[el])+'\n')
    fk2.write(str(k2Norm[el])+'\n')
    

fk1.write('\n')
fk2.write('\n')
fk1.write('VECTORS p1dir float\n')
fk2.write('VECTORS p2dir float\n')
for el in tqdm(np.arange(len(p1))):
    fk1.write(str(p1[el,0])+' '+str(p1[el,1])+' '+str(p1[el,2])+'\n')
    fk2.write(str(p2[el,0])+' '+str(p2[el,1])+' '+str(p2[el,2])+'\n')


fK.close()
fH.close()
fD.close()
fk1.close()
fk2.close()