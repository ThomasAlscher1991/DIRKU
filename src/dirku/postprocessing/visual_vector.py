import skimage.measure
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
def visual_vector(device: str,workingDirectory: str,voxelToMm: Optional[list]=None,segmentsOfInterest: Optional[list]=None,dimension: int=None,slice: int=None):
    """ Plots the displacement vector image.
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
        pts_orig=pts_orig[mask]
    else:
        pts = checkAffine(device, workingDirectory, pts)
        pts = checkNonrigid(device, workingDirectory, pts)

    if dimension is not None and slice is not None:
        fig, ax = plt.subplots(1)
        if dimension==0:
            pts = pts[pts_orig[:,0] == slice]
            pts_orig = pts_orig[pts_orig[:,0] == slice]
            movingImage =movingImage[0,slice,:,:]
            pts1 = pts[:, 1]
            pts2 = pts[:, 2]
            pts = torch.stack((pts1, pts2), dim=1)
            pts_orig1 = pts_orig[:, 1]
            pts_orig2 = pts_orig[:, 2]
            pts_orig = torch.stack((pts_orig1, pts_orig2), dim=1)
            move=pts-pts_orig
            extent = [0, movingImage.size(1) * voxelToMm[2].cpu(), 0, movingImage.size(0)  * voxelToMm[1].cpu()]
            pts_orig[:,0]=pts_orig[:,0]*voxelToMm[1]
            pts_orig[:,1]=pts_orig[:,1]*voxelToMm[2]
            move[:,0]=move[:,0]*voxelToMm[1]
            move[:,1]=move[:,1]*voxelToMm[2]
        elif dimension==1:
            pts = pts[pts_orig[:,1] == slice]
            pts_orig = pts_orig[pts_orig[:,1] == slice]
            movingImage = movingImage[0, :, slice, :]
            pts1 = pts[:, 0]
            pts2 = pts[:, 2]
            pts = torch.stack((pts1, pts2), dim=1)
            pts_orig1 = pts_orig[:, 0]
            pts_orig2 = pts_orig[:, 2]
            pts_orig = torch.stack((pts_orig1, pts_orig2), dim=1)
            move=pts-pts_orig
            extent = [0, movingImage.size(1) * voxelToMm[2].cpu(), 0, movingImage.size(0)  * voxelToMm[0].cpu()]
            pts_orig[:,0]=pts_orig[:,0]*voxelToMm[0]
            pts_orig[:,1]=pts_orig[:,1]*voxelToMm[2]
            move[:,0]=move[:,0]*voxelToMm[0]
            move[:,1]=move[:,1]*voxelToMm[2]

        elif dimension==2:
            pts = pts[pts_orig[:,2] == slice]
            pts_orig = pts_orig[pts_orig[:,2] == slice]
            movingImage = movingImage[0, :, :, slice]
            pts1 = pts[:, 0]
            pts2 = pts[:, 1]
            pts = torch.stack((pts1, pts2), dim=1)
            pts_orig1 = pts_orig[:, 0]
            pts_orig2 = pts_orig[:, 1]
            pts_orig = torch.stack((pts_orig1, pts_orig2), dim=1)
            move=pts-pts_orig
            extent = [0, movingImage.size(1) * voxelToMm[1].cpu(), 0, movingImage.size(0)  * voxelToMm[0].cpu()]
            pts_orig[:,0]=pts_orig[:,0]*voxelToMm[0]
            pts_orig[:,1]=pts_orig[:,1]*voxelToMm[1]
            move[:,0]=move[:,0]*voxelToMm[0]
            move[:,1]=move[:,1]*voxelToMm[1]

        else:
            raise Exception("Postprocessing check dimensions")

        ax.imshow(movingImage.cpu(),extent=extent,origin='lower')
        ax.set_title(f"vector; dim {dimension} slice {slice}")
        ax.quiver(pts_orig[:, 1].cpu(), pts_orig[:, 0].cpu(), move[:, 1].cpu(), move[:, 0].cpu(),color='r', angles='xy', scale_units='xy', scale=1)
        plt.show()
    else:
        extent = [0, movingImage.size(2) * voxelToMm[1].cpu(), 0, movingImage.size(1) * voxelToMm[0].cpu()]
        move = pts - pts_orig
        fig,ax=plt.subplots(1)
        ax.imshow(movingImage[0].cpu(),extent=extent,origin='lower')
        ax.set_title("vector")
        pts_orig[:, 0] = pts_orig[:, 0] * voxelToMm[0]
        pts_orig[:, 1] = pts_orig[:, 1] * voxelToMm[1]
        move[:, 0] = move[:, 0] * voxelToMm[0]
        move[:, 1] = move[:, 1] * voxelToMm[1]
        ax.quiver(pts_orig[:, 1].cpu(), pts_orig[:, 0].cpu(), move[:, 1].cpu(), move[:, 0].cpu(),color='r', angles='xy', scale_units='xy', scale=1)
        plt.show()





def thesis(device,workingDirectory,voxelToMm=None,segmentsOfInterest=None,dimension=None,slice=None):


    #BASICS: load images
    movingImageMask=torch.unsqueeze(torch.from_numpy(np.load(os.path.join(workingDirectory, "moving_mask.npy"))), dim=0).to(device=device)
    movingImage=torch.unsqueeze(torch.from_numpy(np.load(os.path.join(workingDirectory, "moving.npy"))), dim=0).to(device=device)

    indices = np.indices(movingImage.cpu()[0].size())
    pts = np.empty((np.prod(movingImage.cpu().size()), len(movingImage[0].cpu().size())))
    for i, slide in enumerate(indices):
        pts[:, i] = slide.flatten()
    pts=torch.from_numpy(pts).to(device=device).float()
    pts_orig=pts.clone()

    if dimension == 0:
        pts_orig = pts_orig[pts_orig[:, 0] == slice]
    elif dimension == 1:
        pts_orig = pts_orig[pts_orig[:, 1] == slice]
    elif dimension == 2:
        pts_orig = pts_orig[pts_orig[:, 2] == slice]
    pts=pts_orig.clone()




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
    pts_orig=pts_orig[mask]


    fig, ax = plt.subplots(ncols=2)
    movingImage = movingImage[0, :, slice, :]
    movingImageMask=movingImageMask[0, :, slice, :]
    cnts=skimage.measure.find_contours(movingImageMask.cpu().numpy())
    extent = [0, movingImage.size(1) * voxelToMm[2].cpu(), 0, movingImage.size(0) * voxelToMm[0].cpu()]

    ax[0].imshow(movingImage.cpu()*-1,extent=extent,cmap="binary",origin="lower")
    for c in cnts:
        c[:,0]=c[:,0]*voxelToMm[0].cpu().numpy()
        c[:,1]=c[:,1]*voxelToMm[2].cpu().numpy()
        ax[0].plot(c[:,1],c[:,0],c="b",linewidth=5)

    ax[1].imshow(movingImage.cpu()*-1,extent=extent,cmap="binary",origin="lower")
    for c in cnts:

        ax[1].plot(c[:,1],c[:,0],c="b")
    pts_orig0HELP=pts_orig[ptsSegmentation==0]
    pts0HELP=pts[ptsSegmentation==0]

    pts_orig1HELP=pts_orig[ptsSegmentation==1]
    pts1HELP=pts[ptsSegmentation==1]

    pts1 = pts0HELP[:, 0]
    pts2 = pts0HELP[:, 2]
    pts = torch.stack((pts1, pts2), dim=1)
    pts_orig1 = pts_orig0HELP[:, 0]
    pts_orig2 = pts_orig0HELP[:, 2]
    pts_orig = torch.stack((pts_orig1, pts_orig2), dim=1)
    move=pts-pts_orig
    pts_orig[:,0]=pts_orig[:,0]*voxelToMm[0]
    pts_orig[:,1]=pts_orig[:,1]*voxelToMm[2]
    move[:,0]=move[:,0]*voxelToMm[0]
    move[:,1]=move[:,1]*voxelToMm[2]
    ax[0].quiver(pts_orig[:, 1].cpu(), pts_orig[:, 0].cpu(), move[:, 1].cpu(), move[:, 0].cpu(),color='r', angles='xy', scale_units='xy', scale=1, width=0.0075 )

    pts1 = pts1HELP[:, 0]
    pts2 = pts1HELP[:, 2]
    pts = torch.stack((pts1, pts2), dim=1)
    pts_orig1 = pts_orig1HELP[:, 0]
    pts_orig2 = pts_orig1HELP[:, 2]
    pts_orig = torch.stack((pts_orig1, pts_orig2), dim=1)
    move=pts-pts_orig
    pts_orig[:,0]=pts_orig[:,0]*voxelToMm[0]
    pts_orig[:,1]=pts_orig[:,1]*voxelToMm[2]
    move[:,0]=move[:,0]*voxelToMm[0]
    move[:,1]=move[:,1]*voxelToMm[2]
    ax[0].quiver(pts_orig[:, 1].cpu(), pts_orig[:, 0].cpu(), move[:, 1].cpu(), move[:, 0].cpu(),color='g', angles='xy', scale_units='xy', scale=1, width=0.0075 )

    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_ylim([70,130])
    ax[0].set_xlim([240,280])

    from matplotlib.patches import Rectangle
    square = Rectangle((240, 70), 40, 60,
                       linewidth=2, edgecolor='y', facecolor='none',linestyle='--')

    # Add the square to the plot
    ax[1].add_patch(square)



    plt.tight_layout()
    plt.show()

