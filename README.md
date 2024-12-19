# Welcome to DIRKU
The **D**eformable **I**mage **R**egistration **K**øbenhavns **U**niversitet repository contains software developed at the Department of Computer Science at the University of Copenhagen dealing with flow based image registration.

Publications based on this software are:
- pub1
- pub2

# Why use DIRKU?
Flow-based image registratino has some nice porperties. for example they allow you to deal with in its fully developed form with lrage defrmations and diffeomorphisms (LDDMM). with the right adjsutments free form deformations or stationatry velcoity field registration can be acheived.
it contains the most common similarity measuer (NMI, ncc, ssd) and additionally to that mutliple regularizers.
specially developed in concert with simulations, collision detection and self collision detection is part is it.
also we have postprocessing 
#Components
The core concept is widely used in iamge registration.
an energy functional describes the registration problem 


TO DO:
document meshing
document optimization
document postprocessing
create 3d test cases
write sampel scripts
	use stochastic points and optimization
write ci github
show installation
	conda install
	install setup.py
describe teh components
	geometric tranforatmion
        math
        pictures
        material vs spatial confiuguration

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


save folder structure

working directory is the direcotry where all the data is stored for on ecase