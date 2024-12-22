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


def measure_inverseConsistency(device: str,workingDirectory: str,voxelToMm: Optional[list]=None,segmentsOfInterest: Optional[list]=None)->Tuple[Tensor,Tensor]:
    """ Calculates the inverse consistency.
        :param device: sets the computation device, see torch
        :type device: string
        :param workingDirectory: path to working directory, see docs
        :type workingDirectory: string
        :param segmentsOfInterest: segmentations of interest list
        :type segmentsOfInterest: list
        :param voxelSizes: cell dimensions in mm
        :type voxelSizes: torch.Tensor
        :return : mean and standard deviation of inverse consistency
        :rtype : Tensor,Tensor
    """

    #BASICS: load images
    movingImageMask=torch.unsqueeze(torch.from_numpy(np.load(os.path.join(workingDirectory, "moving_mask.npy"))), dim=0).to(device=device)
    movingImage=torch.unsqueeze(torch.from_numpy(np.load(os.path.join(workingDirectory, "moving.npy"))), dim=0).to(device=device)
    indices = np.indices(movingImage.cpu()[0].size())
    pts = np.empty((np.prod(movingImage.cpu().size()), len(movingImage[0].cpu().size())))
    for i, slide in enumerate(indices):
        pts[:, i] = slide.flatten()
    pts=torch.from_numpy(pts).to(device=device).float()
    pts_orig=pts.clone()

    if segmentsOfInterest is not None:
        inter = interpolation.nearest(device,
                                      scale=torch.ones(pts.size(1), device=device))
        ptsSegmentation = inter(pts, movingImageMask).flatten().long()

        mask = torch.zeros_like(ptsSegmentation, dtype=torch.bool)
        for segment in segmentsOfInterest:
            mask |= (ptsSegmentation == segment)
        pts=pts[mask]
        ptsSegmentation=ptsSegmentation[mask]
        pts=checkNonrigid(device,workingDirectory,pts,ptsSegmentation)
        pts=checkNonrigidInverse(device,workingDirectory,pts,ptsSegmentation)
        pts_orig=pts_orig[mask]

    else:
        pts = checkAffine(device, workingDirectory, pts)
        pts = checkNonrigid(device, workingDirectory, pts)
        pts=checkNonrigidInverse(device,workingDirectory,pts)

    if pts.size(1)==3:
        dist=pts-pts_orig
        dist[:,0]=dist[:,0]*voxelToMm[0]
        dist[:,1]=dist[:,1]*voxelToMm[1]
        dist[:,2]=dist[:,2]*voxelToMm[2]
        dist=torch.norm(dist,dim=1)
    else:
        dist=pts-pts_orig
        dist[:,0]=dist[:,0]*voxelToMm[0]
        dist[:,1]=dist[:,1]*voxelToMm[1]
        dist=torch.norm(dist,dim=1)

    return torch.mean(dist),torch.std(dist),dist



