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
def visual_pullback(device: str,workingDirectory: str,voxelToMm: Optional[list]=None,segmentsOfInterest: Optional[list]=None,dimension: int=None,slice: int=None):
    """ Plots the pullback image.
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
    fixedImage=torch.unsqueeze(torch.from_numpy(np.load(os.path.join(workingDirectory, "fixed.npy"))), dim=0).to(device=device)

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

    interpolator=interpolation.cubic(device,torch.ones(pts.size(1),device=device))
    intensitiesMoving,_,_=interpolator(pts_orig,movingImage)
    movingImageResampled=movingImage.clone().float()
    intensitiesFixed,_,_=interpolator(pts_orig,fixedImage)
    fixedImageResampled=fixedImage.clone().float()
    pullback=movingImage.clone()*0.
    intensities,_,_=interpolator(pts,fixedImage)


    if dimension is not None and slice is not None:
        pullback[0,pts_orig[:,0].long(),pts_orig[:,1].long(),pts_orig[:,2].long()]=intensities.flatten()
        movingImageResampled[0,pts_orig[:,0].long(),pts_orig[:,1].long(),pts_orig[:,2].long()]=intensitiesMoving.flatten()
        fixedImageResampled[0,pts_orig[:,0].long(),pts_orig[:,1].long(),pts_orig[:,2].long()]=intensitiesFixed.flatten()

        if dimension==0:
            pullback=pullback[0,slice,:,:]
            movingImageResampled=movingImageResampled[0,slice,:,:]
            fixedImageResampled=fixedImageResampled[0,slice,:,:]
            extent = [0, movingImageResampled.size(1) * voxelToMm[2].cpu(), 0, movingImageResampled.size(0)  * voxelToMm[1].cpu()]

        elif dimension==1:
            pullback=pullback[0,:,slice,:]
            movingImageResampled=movingImageResampled[0,:,slice,:]
            fixedImageResampled=fixedImageResampled[0,:,slice,:]
            extent = [0, movingImageResampled.size(1) * voxelToMm[2].cpu(), 0, movingImageResampled.size(0) * voxelToMm[0].cpu()]


        elif dimension==2:
            pullback=pullback[0,:,:,slice]
            movingImageResampled=movingImageResampled[0,:,:,slice]
            fixedImageResampled=fixedImageResampled[0,:,:,slice]
            extent = [0, movingImageResampled.size(1) * voxelToMm[1].cpu(), 0, movingImageResampled.size(0) * voxelToMm[0].cpu()]


        else:
            raise Exception("postprocessing check dimensions")

        fig,ax=plt.subplots(4)
        ax[0].imshow(movingImageResampled.cpu(),extent=extent,origin='lower')
        ax[0].set_title("movingImageResampled")
        ax[1].imshow(pullback.cpu(),extent=extent,origin='lower')
        ax[1].set_title("pullback")
        ax[2].imshow((movingImageResampled-fixedImageResampled).cpu(),extent=extent,origin='lower')
        ax[2].set_title("movingImageResampled-fixedImageResampled")
        ax[3].imshow((movingImageResampled-pullback).cpu(),extent=extent,origin='lower')
        ax[3].set_title("movingImageResampled-pullback")
        plt.show()
    else:
        pullback[0,pts_orig[:,0].long(),pts_orig[:,1].long()]=intensities.flatten()
        movingImageResampled[0,pts_orig[:,0].long(),pts_orig[:,1].long()]=intensitiesMoving.flatten().float()
        fixedImageResampled[0,pts_orig[:,0].long(),pts_orig[:,1].long()]=intensitiesFixed.flatten().float()
        extent = [0, movingImage.size(2) * voxelToMm[1].cpu(), 0, movingImage.size(1) * voxelToMm[0].cpu()]
        fig,ax=plt.subplots(4)
        ax[0].imshow(movingImageResampled[0].cpu(),extent=extent,origin='lower')
        ax[0].set_title("movingImageResampled")
        ax[1].imshow(pullback[0].cpu(),extent=extent,origin='lower')
        ax[1].set_title("pullback")
        ax[2].imshow((movingImageResampled-fixedImageResampled)[0].cpu(),extent=extent,origin='lower')
        ax[2].set_title("movingImageResampled-fixedImageResampled")
        ax[3].imshow((movingImageResampled-pullback)[0].cpu(),extent=extent,origin='lower')
        ax[3].set_title("movingImageResampled-pullback")
        plt.show()
        plt.imshow((movingImageResampled-fixedImageResampled)[0].cpu()-(movingImageResampled-pullback)[0].cpu())
        plt.show()





