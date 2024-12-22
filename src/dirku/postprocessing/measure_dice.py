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

def measure_dice(device: str,workingDirectory: str,segmentsOfInterest: Optional[list]=None)->Tensor:
    """ Calculates the Dice-Sørensen index based on segmentation masks for pullback images.
        :param device: sets the computation device, see torch
        :type device: str
        :param workingDirectory: path to working directory
        :type workingDirectory: str
        :param segmentsOfInterest: segmentation integers that are to be measured
        :type segmentsOfInterest: list
        :return: DICE
        :rtype: Tensor
    """
    movingImageMask=torch.unsqueeze(torch.from_numpy(np.load(os.path.join(workingDirectory, "moving_mask.npy"))), dim=0).to(device=device)
    fixedImageMask=torch.unsqueeze(torch.from_numpy(np.load(os.path.join(workingDirectory, "fixed_mask.npy"))), dim=0).to(device=device)
    indices = np.indices(movingImageMask.cpu()[0].size())
    pts = np.empty((np.prod(movingImageMask.cpu().size()), len(movingImageMask[0].cpu().size())))
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
        pts=checkAffine(device,workingDirectory,pts,ptsSegmentation)
        pts=checkNonrigid(device,workingDirectory,pts,ptsSegmentation)
        p=pts_orig.clone()
        p[mask]=pts
        pts=p
    else:
        pts=checkAffine(device,workingDirectory,pts)
        pts=checkNonrigid(device,workingDirectory,pts)
    intensityInterpolator = interpolation.nearest(device, torch.ones(pts.size(1), device=device))
    intensities = intensityInterpolator(pts, fixedImageMask)
    pullbackMask=intensities.flatten().reshape(fixedImageMask.size())
    overlap=torch.where(pullbackMask==movingImageMask,1.,0.)
    dice=2*(torch.sum(overlap))/(torch.numel(movingImageMask)+torch.numel(fixedImageMask))
    return dice








