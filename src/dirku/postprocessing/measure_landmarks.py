import warnings

import torch
import numpy as np
import os
from .. import interpolation, geometricTransformations,  numericalIntegration
import re
import matplotlib.pyplot as plt
import pickle
from .postprocessing_utils import *
from typing import Optional, Type, Union, Tuple
from torch import Tensor

def measure_landmarks(device: str,workingDirectory: str,voxelToMm: Optional[list]=None,segmentsOfInterest: Optional[list]=None)->Tuple[Tensor,Tensor]:
    """ Calculates the target registration error.
        :param device: sets the computation device, see torch
        :type device: string
        :param workingDirectory: path to working directory, see docs
        :type workingDirectory: string
        :param segmentsOfInterest: segmentations of interest list
        :type segmentsOfInterest: list
        :param voxelSizes: cell dimensions in mm
        :type voxelSizes: torch.Tensor
        :return : mean and standard deviation of TRE
        :rtype : Tensor,Tensor
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




