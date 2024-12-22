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

def visual_grid(device: str,workingDirectory: str,voxelToMm: Optional[list]=None,segmentsOfInterest: Optional[list]=None,dimension: int=None,slice: int=None):
    """ Plots the deformation grid.
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
        pts=checkAffine(device,workingDirectory,pts,ptsSegmentation)
        pts=checkNonrigid(device,workingDirectory,pts,ptsSegmentation)
        p=pts_orig.clone()
        p[mask]=pts
        pts=p
    else:
        pts = checkAffine(device, workingDirectory, pts)
        pts = checkNonrigid(device, workingDirectory, pts)


    if dimension is not None and slice is not None:
        fig, ax = plt.subplots(1)
        if dimension==0:
            movingDIM = movingImage[0, slice, :, :]
            selector=pts_orig[:,dimension]==slice
            ptsTemp=pts[selector]
            pts_origTemp=pts_orig[selector]
            extent = [0, movingDIM.size(1) * voxelToMm[2].cpu(), 0, movingDIM.size(0)  * voxelToMm[1].cpu()]
            ax.imshow(movingDIM.cpu(), extent=extent, origin='lower')
            for x in range(movingImage.size(2)):
                ptsX=ptsTemp[pts_origTemp[:,1]==x]
                ax.plot(ptsX[:,2].cpu()*voxelToMm[2].cpu(),ptsX[:,1].cpu()*voxelToMm[1].cpu(),c='r')
            for y in range(movingImage.size(3)):
                ptsY=ptsTemp[pts_origTemp[:,2]==y]
                ax.plot(ptsY[:,2].cpu()*voxelToMm[2].cpu(),ptsY[:,1].cpu()*voxelToMm[1].cpu(),c='r')
            plt.show()

        elif dimension==1:
            movingDIM = movingImage[0, :, slice, :]
            selector = pts_orig[:, dimension] == slice
            ptsTemp = pts[selector]
            pts_origTemp = pts_orig[selector]
            extent = [0, movingDIM.size(1) * voxelToMm[2].cpu(), 0, movingDIM.size(0) * voxelToMm[0].cpu()]
            ax.imshow(movingDIM.cpu(), extent=extent, origin='lower')
            for x in range(movingImage.size(1)):
                ptsX = ptsTemp[pts_origTemp[:, 0] == x ]
                ax.plot(ptsX[:, 2].cpu()*voxelToMm[2].cpu(), ptsX[:, 0].cpu()*voxelToMm[0].cpu(), c='r')
            for y in range(movingImage.size(3)):
                ptsY = ptsTemp[pts_origTemp[:, 2] == y]
                ax.plot(ptsY[:, 2].cpu()*voxelToMm[2].cpu(), ptsY[:, 0].cpu()*voxelToMm[0].cpu(), c='r')
            plt.show()
        elif dimension==2:
            movingDIM = movingImage[0, :, :, slice]

            selector = pts_orig[:, dimension] == slice
            ptsTemp = pts[selector]
            pts_origTemp = pts_orig[selector]
            extent = [0, movingDIM.size(1) * voxelToMm[1].cpu(), 0, movingDIM.size(0) * voxelToMm[0].cpu()]
            ax.imshow(movingDIM.cpu(), extent=extent, origin='lower')

            for x in range(movingImage.size(1)):
                ptsX = ptsTemp[pts_origTemp[:, 0] == x]
                ax.plot(ptsX[:, 1].cpu()*voxelToMm[1].cpu(), ptsX[:, 0].cpu()*voxelToMm[0].cpu(), c='r')

            for y in range(movingImage.size(2)):
                ptsY = ptsTemp[pts_origTemp[:, 1] == y]
                ax.plot(ptsY[:, 1].cpu()*voxelToMm[1].cpu(), ptsY[:, 0].cpu()*voxelToMm[0].cpu(), c='r')

            plt.show()
        else:
            print("check dimension")
    else:
        fig, ax = plt.subplots(1)

        movingDIM = movingImage[0]

        extent = [0, movingDIM.size(1) * voxelToMm[1].cpu(), 0, movingDIM.size(0) * voxelToMm[0].cpu()]
        for x in range(movingImage.size(1)):
            ptsX = pts[pts_orig[:, 0] == x]
            ax.plot(ptsX[:, 1].cpu()*voxelToMm[1].cpu(), ptsX[:, 0].cpu()*voxelToMm[0].cpu(), c='r')
        for y in range(movingImage.size(2)):
            ptsY = pts[pts_orig[:, 1] == y]
            ax.plot(ptsY[:, 1].cpu()*voxelToMm[1].cpu(), ptsY[:, 0].cpu()*voxelToMm[0].cpu(), c='r')
        plt.xticks([])  # Removes the x-axis numbers
        plt.yticks([])
        plt.show()



