# Welcome to DIRKU
The **D**eformable **I**mage **R**egistration **KU** repository contains software developed at the Department of Computer Science at the University of Copenhagen dealing with flow based image registration.
# Why use DIRKU?
Flow-based image registration offers several beneficial properties. Notably, it allows for handling large deformations and diffeomorphisms in their fully developed form, such as in Large Deformation Diffeomorphic Metric Mapping (LDDMM). With the proper adjustments, free-form deformations or stationary velocity field registration can also be achieved.

DIRKU supports several commonly used similarity measures, including Normalized Mutual Information (NMI), Normalized Cross-Correlation (NCC), and Sum of Squared Differences (SSD). In addition, it offers multiple regularizers to improve registration accuracy.

Designed with simulations in mind, DIRKU also integrates collision detection and self-collision detection, making it highly suitable for complex image registration tasks.

Moreover, DIRKU includes post-processing capabilities for further refinement.

All functionality is available for 2D and 3D cases.
# How to install
The easiest way to use DIRKU is through Conda, though Git is also required for cloning the repository.
Clone the repository:

 ```git clone https://github.com/ThomasAlscher1991/DIRKU.git```

Navigate to the downloaded directory and enter DIRK/src.
Set up a Conda environment and install the dependencies:

```conda env create -name DIRKU --file requirements.yml ```

Once the environment is set up, navigate to the src folder and install DIRKU:

```pip install .```

# Components
The core concept of DIRKU is built upon widely used techniques in image registration. The problem is formulated using an energy functional, which is minimized to achieve the desired transformation.

The energy functional is expressed as: Φ=arg⁡min⁡Φ∫ΩM(I(X),J(Φ(X)))+S(Φ(X)) dXΦ=argminΦ​∫Ω​M(I(X),J(Φ(X)))+S(Φ(X))dX

## Geometric transformation
To describe the transformation from the moving image to the fixed image, a transformation model is required. Transformations are modeled as affine transformations or non-rigid deformations.

In the context of flow-based registration, we use terminology derived from continuum mechanics. The material configuration refers to the undeformed moving image, while the spatial configuration refers to the deformed moving image.
## Similarity measures
Similarity measures are essential for estimating how similar the transformed moving image is to the fixed image. DIRKU implements three key measures: Sum of Squared Differences (SSD), Normalized Cross-Correlation (NCC), and Normalized Mutual Information (NMI).

## Optimization scheme
To minimize the energy functional and find the optimal transformation, an optimization scheme is used. DIRKU supports two primary optimization techniques: gradient descent and ADMM (Alternating Direction Method of Multipliers). Both methods have stochastic variations.

For ADMM, Eulerian and Lagrangian constraints are used to connect the split energy functional terms. Additionally, a backtracking line search based on the Armijo condition, implemented using PyTorch's optimizer class, is provided.

## Interpolation
Interpolation is necessary for determining image intensities at non-integer locations and for interpolating velocity fields at positions between control points. DIRKU supports several interpolation methods, including:
- Nearest neighbor
- Linear
- Cubic
- 
The same interpolators are used for both images (1 channel) and velocity fields (2 or 3 channels), thanks to a unified approach. Any field passed to the interpolator must have at least one channel. The scale argument defines the control point distances for the interpolation. These interpolators also allow on-the-fly computation of the Jacobian and Laplacian.

## Integration
For time integration, both explicit and implicit methods are available. DIRKU supports the Forward Euler method and the Trapezoidal method, both of which solve the initial value problem encountered in flow-based registrations.

Regularizers or similarity measures can be integrated either at the end of the time interval (as main terms) or at every time step (as regularization terms).

## Regularization
DIRKU provides simple L2 regularization, linear operators, and Saint-Venant Kirchhoff regularization. A major focus is on physics-based constraints, specifically collision detection.

Collision detection is implemented for both inter-collision (between two distinct bodies) and intra-collision (within a single body). Meshes are required for both types of collision detection: 2D triangle meshes for 2D collisions, and 3D tetrahedral meshes for 3D collisions.

One solution for mesh generation is available through the [meshing](https://github.com/ThomasAlscher1991/meshing) repository.

## Folder structure
Users are free to organize their data in any structure they prefer. However, we provide a sample structure that may simplify the process if followed.
Results are saved in 3 ways: a json file for convergences, a pickle for the class and a numpy array for the velocity field.
The working directory is where all the data for a specific case should be stored. Images and landmarks must be loaded as NumPy arrays. The recommended folder structure is as follows:
```
workingDirectory
 ├──results
 │   ├──class_transformation_nonrigid_segment_1_scale_tensor([1., 1.]).pkl
 │   ├──transformation_nonrigid_segment_1_scale_tensor([1., 1.]).json
 │   └──transformation_nonrigid_segment_1_scale_tensor([1., 1.]).npy
 ├──(optional) reuse 
 │   └──sdf_segment_1.npy
 ├──moving.npy
 ├──fixed.npy
 ├──(optional) moving_mask.npy
 ├──(optional) fixed_mask.npy
 ├──(optional) moving_landmarks.npy
 └──(optional) fixed_landmarks.npy
```

PlaceHolder Pictures

## Acknowldegments
To Dorian Depriester for his algorithm on tetrahedron-point-check which inspired our formulation.
