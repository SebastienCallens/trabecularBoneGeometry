# trabecularBoneGeometry
A small collection of tools for the quantification of local and global bone geometry, corresponding to the publication "The local and global geometry of trabecular bone" by S.J.P. Callens et al. (bioRxiv, 2020).

## meshAnalyze
This code takes a mesh (e.g. stl) as input, performs a curvature estimation (using libigl, Panozzo et al.), and writes the curvature-colored mesh to .vtk format so it can easily be visualized in Paraview. Additionally, the mesh vertices, faces and two principal curvatures are stored for later analysis.

## computeISD
This code takes the previously computed curvatures and computes joint propbability distributions (interface shape distribution, see papers from Genau and Voorhees).

## computeRDF
This code takes the previously computed curvatures and computes the radial distribution function of the mean curvature (see papers from Genau and Voorhees). The same can be done for the Gaussian curvature (or, potentially, for a different mesh metric). 

## writePOLY
This code writes the .POLY file of a mesh for analysis using Karambola (https://github.com/morphometry/karambola), an open-source tool for computing the scalar and tensorial Minkowski functionals.
