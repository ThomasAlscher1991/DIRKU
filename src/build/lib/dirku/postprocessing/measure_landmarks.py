import warnings

import torch
import numpy as np
import os
from .. import interpolation, geometricTransformations,  numericalIntegration
import re
import matplotlib.pyplot as plt
import pickle
from .postprocessing_utils import *

def measure_landmarks(device,workingDirectory,voxelToMm=None,segmentsOfInterest=None):
    """ POSTPROCESSING VECTORS
    Sample script to create a vector image.
    Use the same interpolators, integrators, geometric transformations with the same class variables as used in the optimization.
    Set the following variables
        :param device: sets the computation device, see torch
        :type device: string
        :param workingDirectory: path to working directory, see docs
        :type workingDirectory: string
        :param dimension: if 3D registration, set with dimension should be displayed
        :type dimension: int
        :param slice: if 3D registration, set with slice in dimension should be displayed
        :type slice: int
        :param voxelToMm: voxel or pixel size to mm; used to scale the image plot; one entry corresponding to each image dimension;
        :type voxelToMm: torch.tensor
    """
    # BASICS: load images
    movingImageMask = torch.unsqueeze(torch.from_numpy(np.load(os.path.join(workingDirectory, "moving_mask.npy"))),
                                      dim=0).to(device=device)
    landmarkCoordinatesStart = torch.from_numpy(np.load(os.path.join(workingDirectory, "moving_landmarks.npy"))).to(
        device=device).float()
    landmarkCoordinatesEnd = torch.from_numpy(np.load(os.path.join(workingDirectory, "fixed_landmarks.npy"))).to(
        device=device).float()

    if segmentsOfInterest is not None:
        inter = interpolation.nearest(device,
                                      scale=torch.ones(landmarkCoordinatesStart.size(1), device=device))
        lmSegmentation = inter(landmarkCoordinatesStart, movingImageMask).flatten().long()

        mask = torch.zeros_like(lmSegmentation, dtype=torch.bool)
        for segment in segmentsOfInterest:
            mask |= (lmSegmentation == segment)
        landmarkCoordinatesStart=landmarkCoordinatesStart[mask]
        lmSegmentation=lmSegmentation[mask]
        landmarkCoordinatesEnd=landmarkCoordinatesEnd[mask]
        landmarkCoordinatesStart=checkAffine(device,workingDirectory,landmarkCoordinatesStart,lmSegmentation)
        landmarkCoordinatesStart=checkNonrigid(device,workingDirectory,landmarkCoordinatesStart,lmSegmentation)
    else:
        landmarkCoordinatesStart=checkAffine(device,workingDirectory,landmarkCoordinatesStart)
        landmarkCoordinatesStart=checkNonrigid(device,workingDirectory,landmarkCoordinatesStart)

    diff = landmarkCoordinatesStart - landmarkCoordinatesEnd
    diff = diff * voxelToMm
    dist = torch.norm(diff, dim=1)
    print("Mean TRE: ", torch.mean(dist), "STD TRE: ", torch.std(dist))
    diff = torch.round(landmarkCoordinatesStart) - landmarkCoordinatesEnd
    diff = diff * voxelToMm
    dist = torch.norm(diff, dim=1)
    print("fit to closest pixel value approach")
    print("Mean TRE: ", torch.mean(dist), "STD TRE: ", torch.std(dist))

    return torch.mean(dist), torch.std(dist), dist
    # Get solutions
    """files = os.listdir(os.path.join(workingDirectory, "results"))
    filtered_files = [file for file in files if file.startswith("transformation_nonrigid") and file.endswith(".npy")]"""


    """# Apply affine
    if os.path.exists(os.path.join(workingDirectory, "results", "transformation_affine.npy")):
        print("overall affine registration DONE")
        affineMat = torch.from_numpy(
            np.load(os.path.join(workingDirectory, "results", "transformation_affine.npy"))).to(device=device)
        affine = geometricTransformations.affineTransformation(landmarkCoordinatesStart)
        landmarkCoordinatesUpdated = affine.apply(landmarkCoordinatesStart, affineMat)
    else:
        landmarkCoordinatesUpdated = landmarkCoordinatesStart

    for segment in listOfSegments:
        if os.path.exists(os.path.join(workingDirectory, "results", f"transformation_affine_{segment}.npy")):
            print(f" segment affine registration DONE {segment}")

            affineMat = torch.from_numpy(
                np.load(os.path.join(workingDirectory, "results", f"transformation_affine_{segment}.npy"))).to(device=device)
            landmarkCoordinatesUpdatedSegment = landmarkCoordinatesUpdated[lmSegments.flatten() == segment]
            affine = geometricTransformations.affineTransformation(landmarkCoordinatesUpdatedSegment)
            landmarkCoordinatesUpdatedSegment = affine.apply(landmarkCoordinatesUpdatedSegment, affineMat)
            landmarkCoordinatesUpdated[lmSegments.flatten() == segment] = landmarkCoordinatesUpdatedSegment
            if segment in assignedLmSegments:
                assignedLmSegments.remove(segment)
        else:
            pass"""




    # Apply nonrigid
    """if len(filtered_files) > 0:
        assignedLmSegments = torch.unique(lmSegments).tolist()

        sorted_files = sorted(filtered_files,
                              key=lambda s: (extract_segment_and_scale_NPY(s)[0], extract_segment_and_scale_NPY(s)[1]),
                              reverse=True)
        for f in sorted_files:
            velocityField = torch.from_numpy(np.load(os.path.join(workingDirectory, "results", f))).to(device=device)
            scale,segment=extract_segment_and_scale_NPY(f)
            if segment is None:
                print("general deformable registreiton")
                scale = torch.tensor(scale).to(device=device)
                with open(os.path.join(workingDirectory, "results",
                                       "class_transformation_nonrigid_scale_" + str(scale.cpu()) + ".pkl"),
                          'rb') as input_file:
                    nrDeformation = pickle.load(input_file)
                landmarkCoordinatesUpdated = nrDeformation.apply(landmarkCoordinatesUpdated, velocityField)

            else:
                print(f"segment deformable registreiton {segment}")

                landmarkCoordinatesUpdatedSegment = landmarkCoordinatesUpdated[lmSegments.flatten() == segment]
                scale = torch.tensor(scale).to(device=device)
                with open(os.path.join(workingDirectory, "results",
                                       f"class_transformation_nonrigid_segment_{segment}_scale_" + str(
                                           scale.cpu()) + ".pkl"), 'rb') as input_file:
                    nrDeformation = pickle.load(input_file)
                landmarkCoordinatesUpdatedSegment = nrDeformation.apply(landmarkCoordinatesUpdatedSegment, velocityField)
                landmarkCoordinatesUpdated[lmSegments.flatten() == segment] = landmarkCoordinatesUpdatedSegment
                if segment in assignedLmSegments:
                    assignedLmSegments.remove(segment)
    else:
        pass"""


    """if len(assignedLmSegments)>0:
        print(assignedLmSegments)
        warnings.warn(f"Some landmarks have been assigned to segments with not deformation mapping!. Landmarks assigned to segments {assignedLmSegments} are not moved.")
        diff = landmarkCoordinatesUpdated - landmarkCoordinatesEnd
        diff = diff * voxelToMm
        dist = torch.norm(diff, dim=1)
        print("Mean TRE: ", torch.mean(dist), "STD TRE: ", torch.std(dist))
        diff = torch.round(landmarkCoordinatesUpdated) - landmarkCoordinatesEnd
        diff = diff * voxelToMm
        dist = torch.norm(diff, dim=1)
        print("fit to closest pixel value approach")
        print("Mean TRE: ", torch.mean(dist), "STD TRE: ", torch.std(dist))

        mask = torch.ones_like(lmSegments, dtype=torch.bool)
        for segment in assignedLmSegments:
            mask &= (lmSegments != segment)

        diff = landmarkCoordinatesUpdated[mask] - landmarkCoordinatesEnd[mask]
        #diff = landmarkCoordinatesUpdated[lmSegments!=assignedLmSegments[0]] - landmarkCoordinatesEnd[lmSegments!=assignedLmSegments[0]]
        diff = diff * voxelToMm
        dist = torch.norm(diff, dim=1)
        print("Mean TRE without unassigned LMs: ", torch.mean(dist), "STD TRE: ", torch.std(dist))
        diff = torch.round(landmarkCoordinatesUpdated[mask] - landmarkCoordinatesEnd[mask])
        diff = diff * voxelToMm
        dist = torch.norm(diff, dim=1)
        print("fit to closest pixel value approach")
        print("Mean TRE  without unassigned LMs: ", torch.mean(dist), "STD TRE: ", torch.std(dist))

    else:
        diff = landmarkCoordinatesUpdated - landmarkCoordinatesEnd
        diff = diff * voxelToMm
        dist = torch.norm(diff, dim=1)
        print("Mean TRE: ", torch.mean(dist), "STD TRE: ", torch.std(dist))
        diff = torch.round(landmarkCoordinatesUpdated) - landmarkCoordinatesEnd
        diff = diff * voxelToMm
        dist = torch.norm(diff, dim=1)
        print("fit to closest pixel value approach")
        print("Mean TRE: ", torch.mean(dist), "STD TRE: ", torch.std(dist))
    
    return torch.mean(dist), torch.std(dist), dist"""




