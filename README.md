# Welcome to DIRKU
The **D**eformable **I**mage **R**egistration **KU** repository contains software developed at the Department of Computer Science at the University of Copenhagen dealing with flow based image registration.
# Why use DIRKU?
Flow-based image registratino has some nice porperties. for example they allow you to deal with in its fully developed form with lrage defrmations and diffeomorphisms (LDDMM). with the right adjsutments free form deformations or stationatry velcoity field registration can be acheived.
it contains the most common similarity measuer (NMI, ncc, ssd) and additionally to that mutliple regularizers.
specially developed in concert with simulations, collision detection and self collision detection is part is it.
also we have postprocessing 

# How to install
currently the easiest way to use this package is via conda. obviously git is also needed.
clone the repository
 ```git clone https://github.com/ThomasAlscher1991/DIRKU.git```
navigate to your downloaded file and into DIRK/src
set up a conda envrionmanet and install the reuirements
```conda env create -name DIRKU --file requirements.yml ```
navigate to src and install dirku
```pip install .```
#Components
The core concept is widely used in iamge registration.
an energy functional describes the registration problem 
$$\Phi = argmin_{\Phi} \int_{\Omega}M(I(X), J(\Phi(X)) +S(\Phi(X))dX$$

TO DO:
document postprocessing
create 3d test cases
write sampel scripts
	use stochastic points and optimization
write ci github
create lung fissure repo and link to here'
simplifiy stochastic approaches
describe teh components
	geometric tranforatmion
        math
        pictures
        material vs spatial confiuguration
	optimiyation
		admm
			constraints
		armijo

	interpolation
		scale
		fields
        control points
	integration
		main vs reg
        math
        initial value problem
	regularizations
        math
        meaning
	meshing
		should be taken with care
	sim measres
        math
        what are they good for
    collision detection
        math
		requires meshes (2d and 3d) 
			use whatever you want ( for 2d collison we need 2d tringle meshes and for 3d we need tetrahedral meshes)
			one solution developed for this is found in [meshing](https://github.com/ThomasAlscher1991/meshing)


save folder structure
	working directory is the direcotry where all the data is stored for one case
	images and landmarks have to be loaded as numpy arrays
	the folder structure is as follows
		moving.npy
		fixed.npy
		optional	
			moving_mask.npy
			fixed_mask.npy
			moving_landmarks.npy
			fixed_landmarks.npy
	if using the sample scripts, results will be stored in a sub dolfer called results
		names of results will be affine/nonrigid segment number scale 
	if using reuse on some classes (SDF, meshes) the reusable instances are saved in the subfolder reuse



picture examples

# Acknowldegments
To Dorian Depriester for his algorithm on tetrahedron-point-check which inspired our formulation.
