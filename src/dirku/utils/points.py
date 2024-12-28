import matplotlib.pyplot as plt
import torch
from scipy import ndimage
import math
import numpy as np
import skfmm
from ..interpolation import *
from typing import Optional, Type, Union, Tuple
from torch import Tensor

def getEvaluationPoints(device: str,image: Tensor,mask: Optional[Tensor]=None,maskLabel: Optional[Tensor]=None,dilation: Optional[int]=None,exteriorLayers: Optional[int]=None,percentageOfPoints: Optional[float]=None,random: Optional[bool]=None)->Tensor:
    """Returns a set of evaluation points representing the domains to be registered.
    :param device: computation device, see torch docs
    :type device: str
    :param image: moving image of the registration scene
    :type image: torch.Tensor
    :param mask: if only a certain subdomain is registered this mask of moving image is needed
    :type mask: torch.Tensor
    :param maskLabel: if only a certain subdomain is registered the according label is needed
    :type maskLabel: torch.Tensor
    :param dilation: dilating the mask for int steps
    :type dilation: int
    :param exteriorLayers: only this number of exterior layer of the subdomain are considered
    :type exteriorLayers: int
    :param percentageOfPoints: percentage of points selected from the domain (from 0 to 1)
    :type percentageOfPoints: float
    :param random: switch to randomly select the percentage of points
    :type random: bool
    """
    indices = np.indices(image.cpu()[0].size())
    pts = np.empty((np.prod(image.cpu().size()), len(image[0].cpu().size())))
    for i, slide in enumerate(indices):
        pts[:, i] = slide.flatten()
    pts = torch.from_numpy(pts).to(device=device).float()
    if mask is not None:
        if dilation is not None:
            mask = torch.where(mask == maskLabel, 1, 0)
            mask=torch.from_numpy(ndimage.binary_dilation(mask.cpu(),iterations=dilation)).to(device=device)
            mask=torch.where(mask==1,maskLabel,maskLabel-1).to(device=device)
        if exteriorLayers is not None:
            mask = torch.where(mask == maskLabel, 1, 0).to(device=device)
            exteriorLayer = torch.unsqueeze(torch.from_numpy(
                ndimage.binary_erosion(mask[0].cpu().numpy().astype(np.bool_), iterations=exteriorLayers)).to(
                device=device).int(), dim=0)
            mask = mask - exteriorLayer
            mask=torch.where(mask==1,maskLabel,maskLabel-1)
        pts = pts[mask.flatten() == maskLabel]
    if percentageOfPoints is not None:
        numberOfPointsInt = int(pts.size(0) * percentageOfPoints)
        if random:
            random_tensor = torch.randperm(pts.size(0))[:numberOfPointsInt]
            pts = pts[random_tensor]
        else:
            skip = getSkipInterval(pts, numberOfPointsInt)
            pts = pts[::skip]
    return pts

def getSkipInterval(pts: Tensor,number: int)->int:
    """Calculates the necessary interval between evaluation points to achieve the required amount.
        :param pts: all points
        :type pts: torch.Tensor
        :param number: approximate number of evaluation points required
        :type number: int
        :return skip: interval between points
        :rtype skip: int"""
    ptsNumber=pts.size(0)
    skip=math.ceil(ptsNumber/number)
    if skip==0:
        return 1
    else:
        return skip

def getGridPoints(movingImage: Tensor,scale: Tensor,timesteps: int=1)->Tensor:
    """Returns the control point grid for velocity field interpolation.
        :param movingImage: moving image (1,dim1,dim2 (,dim3))
        :type movingImage: torch.Tensor
        :param scale: tensor with stepsize between two consecutive control points in each dimension in pixel
        :type scale: torch.Tensor
        :param timesteps: time steps into which the interval t=[0;1] is divided
        :type timesteps: int
        :return skip: control point grid
        :rtype skip: torch.Tensor"""
    device=movingImage.device
    if scale.size(0)==2:
        x = torch.zeros((timesteps, 2, int(movingImage.size(1) / scale[0]) + 1, int(movingImage.size(2) / scale[1]) + 1))
        return x.to(device=device)
    elif scale.size(0)==3:
        x = torch.zeros((timesteps, 3, int(movingImage.size(1) / scale[0]) + 1, int(movingImage.size(2) / scale[1]) + 1, int(movingImage.size(3) / scale[2]) + 1))
        return x.to(device=device)
    else:
        raise Exception("wrong dimension: scale & moving image")


def assignPoints(device: str,pts: Tensor,mask: Tensor,segments: Tensor,initialValue: Optional[float]=10000)->Tensor:
    """Given a mask, points are assigned to one of the entries in segments. This process either checks whether a point lies inside a subdomain or to which it is closest to.
        :param device: computation device, see torch docs
        :type device: str
            :param pts: points
        :type pts: torch.Tensor
        :param mask: mask (1,dim1,dim2 (,dim3))
        :type mask: torch.Tensor
        :param segments: segments to which points are assigned to
        :type segments: torch.Tensor
        :param initialValue: initial distance value
        :type initialValue: float
        :return s: segmentation
        :rtype s: torch.Tensor"""
    segmentMasks=[]
    for s in segments:
        segmentMasks.append(torch.where(mask == s, 1, 0))
    interTemp = nearest(device, torch.tensor([1., 1., 1.], device=device))
    s = interTemp(pts, mask)
    s = s.flatten()
    set_Elements = set(torch.unique(s).tolist())
    unique_elements = set_Elements.symmetric_difference(segments)
    if len(unique_elements) != 0:
        wrongVals = unique_elements
        inter = linear(device, torch.tensor([1., 1., 1.], device=device))
        sdfs=[]
        for m in segmentMasks:
            m = torch.where(m == 1, -1, 1)
            sdfs.append(torch.unsqueeze(torch.from_numpy(skfmm.distance(m.cpu()[0])).to(device=device), dim=0))
        for val in wrongVals:
            v = pts[s == val]
            decisionMatrix=torch.ones(v.size(0),device=device)*initialValue
            for i,sdf in enumerate(sdfs):
                v1, _, _ = inter(v, sdf)
                decisionMatrix=torch.where(v1.flatten()<decisionMatrix,segments[i],decisionMatrix)
            s[s == val] = decisionMatrix
    return s

