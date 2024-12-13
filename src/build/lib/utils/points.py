import matplotlib.pyplot as plt
import torch
from scipy import ndimage
import math
import numpy as np
import skfmm
from src.dirku import interpolation

def getEvaluationPoints(device,image,mask=None,maskLabel=None,dilation=None,exteriorLayers=None,percentageOfPoints=None,random=False):
    #numberOfPointsPercent from 0 to1
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



def getEvalPointsInt(image,numberOfPoints=None,mask=None,maskLabel=None,dilation=None):
    """Creates a regular grid of evaluation points in the image domain.
        :param image: image
        :type image: torch.tensor
        :param numberOfPoints: approximate number of evaluation points returned
        :type numberOfPoints: int
        :param mask: if evaluation points are only required for a segmented image area, supply segmentation mask
        :type mask: torch.tensor
        :param maskLabel: segmentation label to be considered for evaluation points
        :type maskLabel: torch.tensor
        :return pts: coordinates of evaluation points (#points, dim)
        :rtype pts: torch.tensor"""
    device=image.device
    if len(image.size())==4:
        s_x, s_y,s_z = (torch.meshgrid([torch.arange(0, image.size()[1], 1), torch.arange(0, image.size()[2], 1), torch.arange(0, image.size()[3], 1)]))
        pts = (torch.stack([s_x.flatten(), s_y.flatten(), s_z.flatten()], dim=1).float()).to(device=device)
        if maskLabel is not None:
            if dilation is not None:
                mask=torch.where(mask==maskLabel,1,0)
                mask=torch.from_numpy(ndimage.binary_dilation(mask.cpu(),iterations=dilation)).to(device=device)
                pts = pts[mask.flatten() == 1]
            else:
                pts = pts[mask.flatten() == maskLabel]
        if numberOfPoints is not None:
            skip = getSkipInterval(pts, numberOfPoints)
            pts = pts[::skip]
    elif len(image.size())==3:
        s_x, s_y = (torch.meshgrid([torch.arange(0, image.size()[1], 1), torch.arange(0, image.size()[2], 1)]))
        pts = (torch.stack([s_x.flatten(), s_y.flatten()], dim=1).float()).to(device=device)
        if maskLabel is not None:
            if dilation is not None:
                mask = torch.where(mask == maskLabel, 1, 0)
                mask=torch.from_numpy(ndimage.binary_dilation(mask.cpu(),iterations=dilation)).to(device=device)
                pts = pts[mask.flatten() == 1]
            else:
                pts = pts[mask.flatten() == maskLabel]
        if numberOfPoints is not None:
            skip=getSkipInterval(pts,numberOfPoints)
            pts=pts[::skip]
    else:
        print("wrong dimension")
    return pts.to(device=device)

def getSkipInterval(pts,number):
    """Calculates the neccessary interval between evaluation points to achieve the required amount.
        :param pts: coordinates (#points, dim) of domain points
        :type pts: torch.tensor
        :param number: approximate number of evaluation points returned
        :type number: int
        :return skip: interval between points
        :rtype skip: int"""
    ptsNumber=pts.size(0)
    skip=math.ceil(ptsNumber/number)
    if skip==0:
        return 1
    else:
        return skip

def getGridPoints(movingImage,scale,timesteps=1):
    """Returns the data point grid for velocity field interpolation.
        :param movingImage: moving image (1,dim1,dim2 (,dim3))
        :type movingImage: torch.tensor
        :param scale: tensor with stepsize between two consecutive data points in each dimension in pixel
        :type scale: torch.tensor
        :param timesteps: time steps into which the t=[0;1] is divided
        :type timesteps: int
        :return skip: control point grid
        :rtype skip: torch.tensor"""
    device=movingImage.device
    if scale.size(0)==2:
        x = torch.zeros((timesteps, 2, int(movingImage.size(1) / scale[0]) + 1, int(movingImage.size(2) / scale[1]) + 1))
        return x.to(device=device)
    elif scale.size(0)==3:
        x = torch.zeros((timesteps, 3, int(movingImage.size(1) / scale[0]) + 1, int(movingImage.size(2) / scale[1]) + 1, int(movingImage.size(3) / scale[2]) + 1))
        return x.to(device=device)
    else:
        raise Exception("wrong dimension: scale & moving image")


def getEvalPointsExterior(image,numberOfPoints=None,mask=None,maskLabel=None,dilation=None,layers=1):
    """Creates a regular grid of evaluation points in the image domain.
        :param image: image
        :type image: torch.tensor
        :param numberOfPoints: approximate number of evaluation points returned
        :type numberOfPoints: int
        :param mask: if evaluation points are only required for a segmented image area, supply segmentation mask
        :type mask: torch.tensor
        :param maskLabel: segmentation label to be considered for evaluation points
        :type maskLabel: torch.tensor
        :return pts: coordinates of evaluation points (#points, dim)
        :rtype pts: torch.tensor"""
    device=image.device
    if len(image.size())==4:
        s_x, s_y,s_z = (torch.meshgrid([torch.arange(0, image.size()[1], 1), torch.arange(0, image.size()[2], 1), torch.arange(0, image.size()[3], 1)]))
        pts = (torch.stack([s_x.flatten(), s_y.flatten(), s_z.flatten()], dim=1).float()).to(device=device)
        if maskLabel is not None:
            if dilation is not None:
                mask=torch.where(mask==maskLabel,1,0)
                mask=torch.from_numpy(ndimage.binary_dilation(mask.cpu(),iterations=dilation)).to(device=device).int()
                exteriorLayer = torch.unsqueeze(torch.from_numpy(
                    ndimage.binary_erosion(mask[0].cpu().numpy().astype(np.bool_), iterations=layers)).to(
                    device=device).int(), dim=0)
                mask = mask - exteriorLayer
                pts = pts[mask.flatten() == 1]
            else:
                exteriorLayer = torch.unsqueeze(torch.from_numpy(
                    ndimage.binary_erosion(mask[0].cpu().numpy().astype(np.bool_), iterations=layers)).to(
                    device=device).int(), dim=0)
                mask = mask - exteriorLayer
                pts = pts[mask.flatten() == maskLabel]
        if numberOfPoints is not None:
            skip = getSkipInterval(pts, numberOfPoints)
            pts = pts[::skip]
    elif len(image.size())==3:
        s_x, s_y = (torch.meshgrid([torch.arange(0, image.size()[1], 1), torch.arange(0, image.size()[2], 1)]))
        pts = (torch.stack([s_x.flatten(), s_y.flatten()], dim=1).float()).to(device=device)
        if maskLabel is not None:
            if dilation is not None:
                mask = torch.where(mask == maskLabel, 1, 0)
                mask=torch.from_numpy(ndimage.binary_dilation(mask.cpu().numpy(),iterations=dilation)).to(device=device).int()
                exteriorLayer=torch.unsqueeze(torch.from_numpy(ndimage.binary_erosion(mask[0].cpu().numpy().astype(np.bool_) ,iterations=layers)).to(device=device).int(),dim=0)
                mask=mask-exteriorLayer
                pts = pts[mask.flatten() == 1]
            else:
                exteriorLayer = torch.unsqueeze(torch.from_numpy(
                    ndimage.binary_erosion(mask[0].cpu().numpy().astype(np.bool_), iterations=layers)).to(
                    device=device).int(), dim=0)
                mask = mask - exteriorLayer
                pts = pts[mask.flatten() == maskLabel]
        if numberOfPoints is not None:
            skip=getSkipInterval(pts,numberOfPoints)
            pts=pts[::skip]
    else:
        print("wrong dimension")
    return pts.to(device=device)

def getEvalPointsPercentRandom(image,percent=None,mask=None,maskLabel=None,dilation=None):
    """Creates a regular grid of evaluation points in the image domain.
        :param image: image
        :type image: torch.tensor
        :param numberOfPoints: approximate number of evaluation points returned
        :type numberOfPoints: int
        :param mask: if evaluation points are only required for a segmented image area, supply segmentation mask
        :type mask: torch.tensor
        :param maskLabel: segmentation label to be considered for evaluation points
        :type maskLabel: torch.tensor
        :return pts: coordinates of evaluation points (#points, dim)
        :rtype pts: torch.tensor"""
    device=image.device
    s_x, s_y,s_z = (torch.meshgrid([torch.arange(0, image.size()[1], 1), torch.arange(0, image.size()[2], 1), torch.arange(0, image.size()[3], 1)]))
    pts = (torch.stack([s_x.flatten(), s_y.flatten(), s_z.flatten()], dim=1).float()).to(device=device)
    if maskLabel is not None:
        if dilation is not None:
            mask=torch.where(mask==maskLabel,1,0)
            mask=torch.from_numpy(ndimage.binary_dilation(mask.cpu(),iterations=dilation)).to(device=device)
            pts = pts[mask.flatten() == 1]
        else:
            pts = pts[mask.flatten() == maskLabel]
    if percent is not None:
        length=pts.size(0)
        numberOfPoints=int(length*percent)
        random_tensor = torch.randperm(pts.size(0))[:numberOfPoints]
        pts = pts[random_tensor]
    return pts.to(device=device)

def getEvalPointsPercentFixed(image,percent=None,mask=None,maskLabel=None,dilation=None):
    """Creates a regular grid of evaluation points in the image domain.
        :param image: image
        :type image: torch.tensor
        :param numberOfPoints: approximate number of evaluation points returned
        :type numberOfPoints: int
        :param mask: if evaluation points are only required for a segmented image area, supply segmentation mask
        :type mask: torch.tensor
        :param maskLabel: segmentation label to be considered for evaluation points
        :type maskLabel: torch.tensor
        :return pts: coordinates of evaluation points (#points, dim)
        :rtype pts: torch.tensor"""
    device=image.device
    s_x, s_y,s_z = (torch.meshgrid([torch.arange(0, image.size()[1], 1), torch.arange(0, image.size()[2], 1), torch.arange(0, image.size()[3], 1)]))
    pts = (torch.stack([s_x.flatten(), s_y.flatten(), s_z.flatten()], dim=1).float()).to(device=device)
    if maskLabel is not None:
        if dilation is not None:
            mask=torch.where(mask==maskLabel,1,0)
            mask=torch.from_numpy(ndimage.binary_dilation(mask.cpu(),iterations=dilation)).to(device=device)
            pts = pts[mask.flatten() == 1]
        else:
            pts = pts[mask.flatten() == maskLabel]
    if percent is not None:
        length=pts.size(0)
        numberOfPoints=int(length*percent)
        if numberOfPoints is not None:
            skip = getSkipInterval(pts, numberOfPoints)
            print(skip)
            pts = pts[::skip]
    return pts.to(device=device)


def assignPoints(device,pts,mask,segments,initialValue=10000):
    segmentMasks=[]
    for s in segments:
        segmentMasks.append(torch.where(mask == s, 1, 0))
    interTemp = interpolation.nearest(device, torch.tensor([1., 1., 1.], device=device))
    s = interTemp(pts, mask)
    s = s.flatten()
    set_Elements = set(torch.unique(s).tolist())
    unique_elements = set_Elements.symmetric_difference(segments)
    if len(unique_elements) != 0:
        wrongVals = unique_elements
        inter = interpolation.linear(device, torch.tensor([1., 1., 1.], device=device))
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

