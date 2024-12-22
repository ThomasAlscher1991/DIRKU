import torch
import numpy as np
from scipy import ndimage
import os
from .. import  geometricTransformations, interpolation
import pickle
from .postprocessing_utils import *
import math
from typing import Optional, Type, Union, Tuple
from torch import Tensor


def measure_jacobian(device: str,workingDirectory: str,voxelToMm: Optional[Tensor]=None,segmentsOfInterest: Optional[list]=None)->dict:
    """ Calculates the jacobian of teh deformation field.
        :param device: sets the computation device, see torch
        :type device: string
        :param workingDirectory: path to working directory, see docs
        :type workingDirectory: string
        :param segmentsOfInterest: segmentations of interest list
        :type segmentsOfInterest: list
        :param voxelSizes: cell dimensions in mm
        :type voxelSizes: torch.Tensor
        :return fields: dictionary of deformation field jacobians
        :rtype fields: dict
    """
    movingImageMask=torch.unsqueeze(torch.from_numpy(np.load(os.path.join(workingDirectory, "moving_mask.npy"))), dim=0).to(device=device)
    fixedImageMask=torch.unsqueeze(torch.from_numpy(np.load(os.path.join(workingDirectory, "fixed_mask.npy"))), dim=0).to(device=device)
    indices = np.indices(movingImageMask.cpu()[0].size())
    pts = np.empty((np.prod(movingImageMask.cpu().size()), len(movingImageMask[0].cpu().size())))
    for i, slide in enumerate(indices):
        pts[:, i] = slide.flatten()
    pts=torch.from_numpy(pts).to(device=device).float()
    fields={}
    if pts.size(1)==3:
        if segmentsOfInterest is not None:
            for segment in segmentsOfInterest:
                ptsSegmentation = torch.ones(pts.size(0), dtype=torch.bool)*segment
                pts=checkAffine(device,workingDirectory,pts,ptsSegmentation)
                pts=checkNonrigid(device,workingDirectory,pts,ptsSegmentation)

                phiX = (pts[:, 0]).flatten().reshape(fixedImageMask.size())[0] * voxelToMm[0]
                phiY = (pts[:, 1]).flatten().reshape(fixedImageMask.size())[0] * voxelToMm[1]
                phiZ = (pts[:, 2]).flatten().reshape(fixedImageMask.size())[0] * voxelToMm[2]

                xphiX, yphiX, zphiX = compute_gradient_central_diff3D(phiX)
                xphiY, yphiY, zphiY = compute_gradient_central_diff3D(phiY)
                xphiZ, yphiZ, zphiZ = compute_gradient_central_diff3D(phiZ)

                id = torch.eye(3)
                id = id.reshape((1, 3, 3))
                id = id.repeat(phiX.size(0), phiX.size(1), phiX.size(2), 1, 1)

                jacPhi = torch.zeros(id.size())
                jacPhi[:, :, :, 0, 0] = xphiX
                jacPhi[:, :, :, 0, 1] = yphiX
                jacPhi[:, :, :, 0, 2] = zphiX
                jacPhi[:, :, :, 1, 0] = xphiY
                jacPhi[:, :, :, 1, 1] = yphiY
                jacPhi[:, :, :, 1, 2] = zphiY
                jacPhi[:, :, :, 2, 0] = xphiZ
                jacPhi[:, :, :, 2, 1] = yphiZ
                jacPhi[:, :, :, 2, 2] = zphiZ
                fields[str(segment)]=torch.linalg.det(jacPhi).cpu().numpy().tolist()

        else:
            pts=checkAffine(device,workingDirectory,pts)
            pts=checkNonrigid(device,workingDirectory,pts)

            phiX = (pts[:, 0]).flatten().reshape(fixedImageMask.size())[0] * voxelToMm[0]
            phiY = (pts[:, 1]).flatten().reshape(fixedImageMask.size())[0] * voxelToMm[1]
            phiZ = (pts[:, 2]).flatten().reshape(fixedImageMask.size())[0] * voxelToMm[2]

            xphiX, yphiX, zphiX = compute_gradient_central_diff3D(phiX)
            xphiY, yphiY, zphiY = compute_gradient_central_diff3D(phiY)
            xphiZ, yphiZ, zphiZ = compute_gradient_central_diff3D(phiZ)

            id = torch.eye(3)
            id = id.reshape((1, 3, 3))
            id = id.repeat(phiX.size(0), phiX.size(1), phiX.size(2), 1, 1)

            jacPhi = torch.zeros(id.size())
            jacPhi[:, :, :, 0, 0] = xphiX
            jacPhi[:, :, :, 0, 1] = yphiX
            jacPhi[:, :, :, 0, 2] = zphiX
            jacPhi[:, :, :, 1, 0] = xphiY
            jacPhi[:, :, :, 1, 1] = yphiY
            jacPhi[:, :, :, 1, 2] = zphiY
            jacPhi[:, :, :, 2, 0] = xphiZ
            jacPhi[:, :, :, 2, 1] = yphiZ
            jacPhi[:, :, :, 2, 2] = zphiZ
            fields["overall"] = torch.linalg.det(jacPhi).cpu().numpy().tolist()

    else:
        if segmentsOfInterest is not None:
            for segment in segmentsOfInterest:
                ptsSegmentation = torch.ones(pts.size(0), dtype=torch.bool) * segment
                pts = checkAffine(device, workingDirectory, pts, ptsSegmentation)
                pts = checkNonrigid(device, workingDirectory, pts, ptsSegmentation)

                phiX = (pts[:, 0]).flatten().reshape(fixedImageMask.size())[0] * voxelToMm[0]
                phiY = (pts[:, 1]).flatten().reshape(fixedImageMask.size())[0] * voxelToMm[1]

                xphiX, yphiX = compute_gradient_central_diff2D(phiX)
                xphiY, yphiY = compute_gradient_central_diff2D(phiY)

                id = torch.eye(2)
                id = id.reshape((1, 2, 2))
                id = id.repeat(phiX.size(0), phiX.size(1), 1, 1)

                jacPhi = torch.zeros(id.size())
                jacPhi[:, :, 0, 0] = xphiX
                jacPhi[:, :, 0, 1] = yphiX
                jacPhi[:, :, 1, 0] = xphiY
                jacPhi[:, :, 1, 1] = yphiY
                fields[str(segment)] = torch.linalg.det(jacPhi).cpu().numpy().tolist()

        else:
            pts = checkAffine(device, workingDirectory, pts)
            pts = checkNonrigid(device, workingDirectory, pts)

            phiX = (pts[:, 0]).flatten().reshape(fixedImageMask.size())[0] * voxelToMm[0]
            phiY = (pts[:, 1]).flatten().reshape(fixedImageMask.size())[0] * voxelToMm[1]

            xphiX, yphiX = compute_gradient_central_diff2D(phiX)
            xphiY, yphiY = compute_gradient_central_diff2D(phiY)

            id = torch.eye(2)
            id = id.reshape((1, 2, 2))
            id = id.repeat(phiX.size(0), phiX.size(1), 1, 1)

            jacPhi = torch.zeros(id.size())
            jacPhi[:, :, 0, 0] = xphiX
            jacPhi[:, :, 0, 1] = yphiX
            jacPhi[:, :, 1, 0] = xphiY
            jacPhi[:, :, 1, 1] = yphiY
            fields["overall"] = torch.linalg.det(jacPhi).cpu().numpy().tolist()
    return fields











