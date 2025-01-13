import itertools

import skimage.measure
from shapely.geometry import Polygon
from skimage import measure
import torch
import numpy as np
import os
from ..import interpolation, geometricTransformations,  numericalIntegration, utils
import re
import matplotlib.pyplot as plt
import igl
import pickle
from .postprocessing_utils import *
from typing import Optional, Type, Union, Tuple
from torch import Tensor
def measure_intersection3d(device: str,workingDirectory: str,voxelToMm: Optional[Tensor]=None,segmentsOfInterest: Optional[list]=None,vertexLists: list=[],simplexLists: list=[])->Tuple[Tensor,Tensor]:
    """ Calculates the overlap and gap between segmentations in 3D. Objects are input as vertex and simplex collections.
        :param device: sets the computation device, see torch
        :type device: string
        :param workingDirectory: path to working directory, see docs
        :type workingDirectory: string
        :param segmentsOfInterest: segmentations of interest list
        :type segmentsOfInterest: list
        :param voxelToMm: cell dimensions in mm
        :type voxelToMm: torch.Tensor
        :param vertexLists: list of vertex collections
        :type vertexLists: list
        :param simplexLists: list of simplex collections
        :type simplexLists: list
        :return : gap and overlap
        :rtype : Tensor,Tensor
    """
    movingImage = torch.unsqueeze(torch.from_numpy(np.load(os.path.join(workingDirectory, "moving.npy"))), dim=0).to(
        device=device)
    indices = np.indices(movingImage.cpu()[0].size())
    coords = np.empty((np.prod(movingImage.cpu().size()), len(movingImage[0].cpu().size())))
    for i, slide in enumerate(indices):
        coords[:, i] = slide.flatten()
    coords = torch.from_numpy(coords).to(device=device).float()

    intersectionmap = torch.ones(movingImage[0].size(), device=device) * -1

    for i,segment in enumerate(segmentsOfInterest):
        vertices=vertexLists[i]
        simplices=simplexLists[i]
        ptsSegmentation = torch.ones(vertices.size(0), dtype=torch.bool)*segment
        vertices = checkAffine(device, workingDirectory, vertices, ptsSegmentation)
        vertices = checkNonrigid(device, workingDirectory, vertices, ptsSegmentation)

        p = igl.signed_distance(coords.cpu().numpy(), vertices.cpu().numpy(), simplices.long().cpu().numpy())
        sdf = movingImage[0].clone().float().to(device=device)
        sdf[coords[:, 0].long(), coords[:, 1].long(), coords[:, 2].long()] = torch.from_numpy(p[0]).flatten().to(
            device=device)
        sdf = torch.where(sdf < 0, 1, 0)
        intersectionmap = intersectionmap + sdf
    if voxelToMm is not None:
        return torch.sum(torch.where(intersectionmap == -1, 1, 0)) * voxelToMm[0] * voxelToMm[1] * voxelToMm[
        2], torch.sum(torch.where(intersectionmap >= 1, 1, 0)) * voxelToMm[0] * voxelToMm[1] * voxelToMm[2]
    else:
        return torch.sum(torch.where(intersectionmap == -1, 1, 0)) , torch.sum(torch.where(intersectionmap >= 1, 1, 0))

def measure_intersection2d(device: str,workingDirectory: str,segmentsOfInterest: Optional[list]=None,contours: list=[])->Tensor:
    """ Calculates the overlap between segmentations in 3D. Objects are input as contour collections.
        :param device: sets the computation device, see torch
        :type device: string
        :param workingDirectory: path to working directory, see docs
        :type workingDirectory: string
        :param segmentsOfInterest: segmentations of interest list
        :type segmentsOfInterest: list
        :param contours: contour collections
        :type contours: list
        :return : overlap
        :rtype : Tensor
    """
    movingImageMask=torch.unsqueeze(torch.from_numpy(np.load(os.path.join(workingDirectory, "moving_mask.npy"))), dim=0).to(device=device)
    indices = np.indices(movingImageMask.cpu()[0].size())
    coords = np.empty((np.prod(movingImageMask.cpu().size()), len(movingImageMask[0].cpu().size())))
    for i, slide in enumerate(indices):
        coords[:, i] = slide.flatten()
    contours=[]
    for i,segment in enumerate(segmentsOfInterest):
        contour=skimage.measure.find_contours(torch.where(movingImageMask[0]==segment,1,0).cpu().numpy(),level=0.5)[0]
        contour=torch.from_numpy(contour).to(device=device)
        ptsSegmentation = torch.ones(contour.size(0), dtype=torch.bool) * segment
        contour = checkAffine(device, workingDirectory, contour, ptsSegmentation)
        contour = checkNonrigid(device, workingDirectory, contour, ptsSegmentation)
        contours.append(contour)
    contoursIndices = np.arange(len(contours))
    combinations = list(itertools.combinations(contoursIndices, 2))
    dictIntersection={}
    for c in combinations:
        polygon1 = Polygon(contours[c[0]].cpu().numpy())
        polygon2 = Polygon(contours[c[1]].cpu().numpy())
        intersection = polygon1.intersection(polygon2)
        intersection_area = intersection.area
        dictIntersection[str(c)]=intersection_area
    return dictIntersection










