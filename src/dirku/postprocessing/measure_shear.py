import torch
import numpy as np
from scipy import ndimage
import os
from .. import  geometricTransformations,interpolation
import matplotlib.pyplot as plt
import pickle
import itertools
import scipy
from .postprocessing_utils import *
from typing import Optional, Type, Union, Tuple
from torch import Tensor
def measure_shear(device: str,workingDirectory: str,voxelToMm: Optional[list]=None,segmentsOfInterest: Optional[list]=None)->dict:
    """ Calculates the maximum shear stretch.
        :param device: sets the computation device, see torch
        :type device: string
        :param workingDirectory: path to working directory, see docs
        :type workingDirectory: string
        :param segmentsOfInterest: segmentations of interest list
        :type segmentsOfInterest: list
        :param voxelSizes: cell dimensions in mm
        :type voxelSizes: torch.Tensor
        :return shearBoundariesDict: dictionary of shear between segmentations
        :rtype shearBoundariesDict: dict
    """
    # BASICS: load images
    movingImageMask = torch.unsqueeze(torch.from_numpy(np.load(os.path.join(workingDirectory, "moving_mask.npy"))),
                                      dim=0).to(device=device)
    movingImage = torch.unsqueeze(torch.from_numpy(np.load(os.path.join(workingDirectory, "moving.npy"))), dim=0).to(
        device=device)

    indices = np.indices(movingImage.cpu()[0].size())
    pts = np.empty((np.prod(movingImage.cpu().size()), len(movingImage[0].cpu().size())))
    for i, slide in enumerate(indices):
        pts[:, i] = slide.flatten()
    pts = torch.from_numpy(pts).to(device=device).float()
    pts_orig = pts.clone()

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
        pts = checkAffine(device, workingDirectory, pts)
        pts = checkNonrigid(device, workingDirectory, pts)

    if pts.size(1)==3:
        disX = (pts[:, 0] - pts_orig[:, 0]).flatten().reshape(movingImageMask.size())[0] * voxelToMm[0]
        disY = (pts[:, 1] - pts_orig[:, 1]).flatten().reshape(movingImageMask.size())[0] * voxelToMm[1]
        disZ = (pts[:, 2] - pts_orig[:, 2]).flatten().reshape(movingImageMask.size())[0] * voxelToMm[2]

        xdisX, ydisX, zdisX = compute_gradient_central_diff3D(disX)
        xdisY, ydisY, zdisY = compute_gradient_central_diff3D(disY)
        xdisZ, ydisZ, zdisZ = compute_gradient_central_diff3D(disZ)
        id = torch.eye(3)
        id = id.reshape((1, 3, 3))
        id = id.repeat(disX.size(0), disX.size(1), disX.size(2), 1, 1)

        graddis = torch.zeros(id.size())
        graddis[:, :, :, 0, 0] = xdisX
        graddis[:, :, :, 0, 1] = ydisX
        graddis[:, :, :, 0, 2] = zdisX
        graddis[:, :, :, 1, 0] = xdisY
        graddis[:, :, :, 1, 1] = ydisY
        graddis[:, :, :, 1, 2] = zdisY
        graddis[:, :, :, 2, 0] = xdisZ
        graddis[:, :, :, 2, 1] = ydisZ
        graddis[:, :, :, 2, 2] = zdisZ
        graddisT = torch.transpose(graddis, dim0=3, dim1=4)
        FtF = id + graddis + graddisT + torch.matmul(graddisT, graddis)
        eigenval = torch.linalg.eigvals(FtF)
        term1 = torch.sqrt(eigenval)
        # reduce imaginary to real numbers
        eigenval = torch.sqrt(term1.real ** 2 + term1.imag ** 2)
        max, _ = torch.max(eigenval, dim=3)
        min, _ = torch.min(eigenval, dim=3)
        shear = (max - min) / 2

        shearBoundariesDict={}
        shearBoundariesDict["overall shear"]=shear.flatten()

        elements = torch.unique(movingImageMask)
        result = list(itertools.combinations(elements, 2))

        for r in result:
            temp1 = torch.where(movingImageMask == r[0], 1, 0)
            temp2 = torch.where(movingImageMask == r[1], 1, 0)
            temp1 = torch.from_numpy(scipy.ndimage.binary_dilation(temp1.cpu(), iterations=1)).to(device=device).int()
            temp2 = torch.from_numpy(scipy.ndimage.binary_dilation(temp2.cpu(), iterations=1)).to(device=device).int()

            temp = temp1 + temp2
            dec = torch.where(temp == 2, 1, 0).to(device=device)
            if torch.sum(dec) > 0:
                print(f"shared boundary between {r}: ", torch.sum(dec))
                shearBoundaries = shear.flatten()[dec.flatten().cpu() == 1]
                meanShearBoundaries = torch.mean(shearBoundaries)
                stdShearBoundaries = torch.std(shearBoundaries)
                print("|mean shear boundaries| ", meanShearBoundaries, "|")
                print("|std shear boundaries |", stdShearBoundaries, "|")
                shearBoundariesDict[r]=shearBoundaries

        return shearBoundariesDict
    else:
        disX = (pts[:, 0] - pts_orig[:, 0]).flatten().reshape(movingImageMask.size())[0] * voxelToMm[0]
        disY = (pts[:, 1] - pts_orig[:, 1]).flatten().reshape(movingImageMask.size())[0] * voxelToMm[1]

        xdisX, ydisX = compute_gradient_central_diff2D(disX)
        xdisY, ydisY = compute_gradient_central_diff2D(disY)
        id = torch.eye(2)
        id = id.reshape((1, 2, 2))
        id = id.repeat(disX.size(0), disX.size(1), 1, 1)

        graddis = torch.zeros(id.size())
        graddis[:, :, 0, 0] = xdisX
        graddis[:, :, 0, 1] = ydisX
        graddis[:, :, 1, 0] = xdisY
        graddis[:, :, 1, 1] = ydisY
        graddisT = torch.transpose(graddis, dim0=2, dim1=3)
        FtF = id + graddis + graddisT + torch.matmul(graddisT, graddis)
        eigenval = torch.linalg.eigvals(FtF)
        term1 = torch.sqrt(eigenval)
        # reduce imaginary to real numbers
        eigenval = torch.sqrt(term1.real ** 2 + term1.imag ** 2)
        max, _ = torch.max(eigenval, dim=2)
        min, _ = torch.min(eigenval, dim=2)
        shear = (max - min) / 2

        shearBoundariesDict={}
        shearBoundariesDict["overall shear"]=shear.flatten()

        elements = torch.unique(movingImageMask)
        tuple_length = 2
        result = list(itertools.combinations(elements, 2))

        for r in result:
            temp1 = torch.where(movingImageMask == r[0], 1, 0)
            temp2 = torch.where(movingImageMask == r[1], 1, 0)
            temp1 = torch.from_numpy(scipy.ndimage.binary_dilation(temp1.cpu(), iterations=1)).to(device=device).int()
            temp2 = torch.from_numpy(scipy.ndimage.binary_dilation(temp2.cpu(), iterations=1)).to(device=device).int()

            temp = temp1 + temp2
            dec = torch.where(temp == 2, 1, 0).to(device=device)
            if torch.sum(dec) > 0:
                print(f"shared boundary between {r}: ", torch.sum(dec))
                shearBoundaries = shear.flatten()[dec.flatten().cpu() == 1]
                meanShearBoundaries = torch.mean(shearBoundaries)
                stdShearBoundaries = torch.std(shearBoundaries)
                print("|mean shear boundaries| ", meanShearBoundaries, "|")
                print("|std shear boundaries |", stdShearBoundaries, "|")

        return shearBoundariesDict









