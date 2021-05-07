# -*- coding: utf-8 -*-
"""
Write poly file for karambola, after meshlab processing (to remove nm vertices))
Developed by Sebastien Callens
"""
# %%
import numpy as np
import trimesh as tr
import igl
import time
from tqdm import tqdm

# %%
#Import mesh. IMPORTANT: make sure it does not contain non-manifold features. 
# This can be verified and adjusted in Meshlab
nameStr = 'test'
meshProcName = 'test.ply'
meshProc = tr.load(meshProcName)
meshverts = meshProc.vertices.view(np.ndarray)
meshfaces = meshProc.faces.view(np.ndarray)
# Using ligigl, detect boundary faces
facesBoundary_igl = igl.boundary_facets(meshfaces)
edgeList = np.unique(meshfaces)
edgeTopology = igl.edge_topology(meshverts,meshfaces)
# edgeTopology[2] is a matrix #e x w with edge-triangle relations. -1 indicates boundary
edgesTri = edgeTopology[2]
# We check wether we are dealing with a -1 or not
boolBoundaryEdges = edgesTri==-1
# We check every row of the 2D array boolBoundaryEdges to see if it contains a "True" (i.e. boundary)
boolBoundaryEdges = boolBoundaryEdges.any(axis=1)
# Select the rows that have a -1
facesBoundary = edgesTri[boolBoundaryEdges,:]
# We want the numbers in every row that are not equal to -1 --> sum every row and do +1
facesBoundary = np.unique(np.sum(facesBoundary,axis=1)+1)
# %%
# Write poly file for karambola
meshfaces_karambola = meshfaces+1
facesBoundary_karambola = facesBoundary+1
karambolaStr = 'batch files/'+nameStr+'.poly'
t = time.time()
counter = 0
f = open(karambolaStr,'w')
f.write('POINTS\n')
for row in tqdm(meshverts):
    counter = counter+1
    f.write(str(counter)+': ')
    f.write(' '.join(str(item) for item in row)+'\n')
counter = 0
f.write('POLYS\n')
for row in tqdm(meshfaces_karambola):
    counter = counter+1
    f.write(str(counter)+': ')
    if np.isin(counter,facesBoundary_karambola):
        f.write(' '.join(str(item) for item in row)+ ' < c(0 , 0 , 0 , 1)'+'\n')
    else:
        f.write(' '.join(str(item) for item in row)+ ' < c(0 , 0 , 0 , 2)'+'\n')
f.write('END')
f.close()
t_elapsed = time.time()-t
print('Elapsed time = ',t_elapsed,' sec')