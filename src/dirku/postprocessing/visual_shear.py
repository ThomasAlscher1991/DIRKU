import torch
import numpy as np
from scipy import ndimage
import os
from .. import  geometricTransformations, interpolation
import matplotlib.pyplot as plt
import pickle
import itertools
import scipy
from .postprocessing_utils import *
from typing import Optional, Type, Union, Tuple
from torch import Tensor
def visual_shear(device: str,workingDirectory: str,voxelToMm: Optional[list]=None,segmentsOfInterest: Optional[list]=None,dimension: int=None,slice: int=None):
    """ Plots the shear image.
        :param device: sets the computation device, see torch
        :type device: string
        :param workingDirectory: path to working directory, see docs
        :type workingDirectory: string
        :param segmentsOfInterest: segmentations of interest list
        :type segmentsOfInterest: list
        :param voxelSizes: cell dimensions in mm
        :type voxelSizes: torch.Tensor
        :param dimension: if 3D registration, set with dimension should be displayed
        :type dimension: int
        :param slice: if 3D registration, set with slice in dimension should be displayed
        :type slice: int
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

        fig, ax = plt.subplots(1)
        if dimension == 0:
            shearDIM = shear[slice, :, :]
            extent = [0, shearDIM.size(1) * voxelToMm[2].cpu(), 0, shearDIM.size(0) * voxelToMm[1].cpu()]
        elif dimension == 1:
            shearDIM = shear[:, slice, :]
            extent = [0, shearDIM.size(1) * voxelToMm[2].cpu(), 0, shearDIM.size(0) * voxelToMm[0].cpu()]
        elif dimension == 2:
            shearDIM = shear[:, :, slice]
            extent = [0, shearDIM.size(1) * voxelToMm[1].cpu(), 0, shearDIM.size(0) * voxelToMm[0].cpu()]
        else:
            raise Exception("postprocessing check dimensions")
        ax.imshow(shearDIM.cpu(), extent=extent, origin='lower')
        ax.set_title(f"shear; dim {dimension} slice {slice}")
        plt.show()

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



        fig, ax = plt.subplots(1)
        extent = [0, shear.size(1) * voxelToMm[1].cpu(), 0, shear.size(0) * voxelToMm[0].cpu()]
        ax.imshow(shear.cpu(), extent=extent, origin='lower')
        ax.set_title(f"shear; dim {dimension} slice {slice}")

        plt.show()





