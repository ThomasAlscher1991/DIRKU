import itertools
from shapely.geometry import Polygon
from skimage import measure
import torch
import numpy as np
import os
from ..import interpolation, geometricTransformations,  numericalIntegration, utils,meshing
import re
import matplotlib.pyplot as plt
import igl
import pickle
from .postprocessing_utils import *

def measure_intersection3d(device,workingDirectory,voxelToMm=None,segmentsOfInterest=None,boundarySub=2):
    # BASICS: load images
    movingImageMask = torch.unsqueeze(torch.from_numpy(np.load(os.path.join(workingDirectory, "moving_mask.npy"))),
                                      dim=0).to(device=device)
    movingImage = torch.unsqueeze(torch.from_numpy(np.load(os.path.join(workingDirectory, "moving.npy"))), dim=0).to(
        device=device)

    indices = np.indices(movingImage.cpu()[0].size())
    coords = np.empty((np.prod(movingImage.cpu().size()), len(movingImage[0].cpu().size())))
    for i, slide in enumerate(indices):
        coords[:, i] = slide.flatten()
    coords = torch.from_numpy(coords).to(device=device).float()

    intersectionmap = torch.ones(movingImage[0].size(), device=device) * -1

    for segment in segmentsOfInterest:
        maskTemp = movingImageMask.clone()
        maskTemp = torch.where(maskTemp == segment, 1, 0)
        maskTemp[:, 0, :, :] = -1
        maskTemp[:, -1, :, :] = -1
        maskTemp[:, :, :, 0] = -1
        maskTemp[:, :, :, -1] = -1
        maskTemp[:, :, 0, :] = -1
        maskTemp[:, :, -1, :] = -1
        mesher = meshing.surfaceMesh(maskTemp[0].cpu(), segment, device, workingDirectory, faceNumbers=5000, reuse=True,
                                     level=0.5)
        pts, simplices = mesher.getVerticesAndSimplicesSurface()
        ptsSegmentation = torch.ones(pts.size(0), dtype=torch.bool)*segment
        pts = checkAffine(device, workingDirectory, pts, ptsSegmentation)
        pts = checkNonrigid(device, workingDirectory, pts, ptsSegmentation)

        p = igl.signed_distance(coords.cpu().numpy(), pts.cpu().numpy(), simplices.long().cpu().numpy())
        sdf = movingImage[0].clone().float().to(device=device)
        sdf[coords[:, 0].long(), coords[:, 1].long(), coords[:, 2].long()] = torch.from_numpy(p[0]).flatten().to(
            device=device)
        sdf = torch.where(sdf < 0, 1, 0)
        intersectionmap = intersectionmap + sdf

    intersectionmap = intersectionmap[boundarySub:-boundarySub, boundarySub:-boundarySub, boundarySub:-boundarySub]
    return torch.sum(torch.where(intersectionmap == -1, 1, 0)) * voxelToMm[0] * voxelToMm[1] * voxelToMm[
        2], torch.sum(torch.where(intersectionmap >= 1, 1, 0)) * voxelToMm[0] * voxelToMm[1] * voxelToMm[2]


def measure_intersection3d_tensor(device,workingDirectory,voxelToMm=None,segmentsOfInterest=None,boundarySub=2):
    # BASICS: load images
    movingImageMask = torch.unsqueeze(torch.from_numpy(np.load(os.path.join(workingDirectory, "moving_mask.npy"))),
                                      dim=0).to(device=device)
    movingImage = torch.unsqueeze(torch.from_numpy(np.load(os.path.join(workingDirectory, "moving.npy"))), dim=0).to(
        device=device)

    indices = np.indices(movingImage.cpu()[0].size())
    coords = np.empty((np.prod(movingImage.cpu().size()), len(movingImage[0].cpu().size())))
    for i, slide in enumerate(indices):
        coords[:, i] = slide.flatten()
    coords = torch.from_numpy(coords).to(device=device).float()

    intersectionmap = torch.ones(movingImage[0].size(), device=device) * -1

    for segment in segmentsOfInterest:
        maskTemp = movingImageMask.clone()
        maskTemp = torch.where(maskTemp == segment, 1, 0)
        maskTemp[:, 0, :, :] = -1
        maskTemp[:, -1, :, :] = -1
        maskTemp[:, :, :, 0] = -1
        maskTemp[:, :, :, -1] = -1
        maskTemp[:, :, 0, :] = -1
        maskTemp[:, :, -1, :] = -1
        mesher = meshing.surfaceMesh(maskTemp[0].cpu(), segment, device, workingDirectory, faceNumbers=5000, reuse=True,
                                     level=0.5)
        pts, simplices = mesher.getVerticesAndSimplicesSurface()
        ptsSegmentation = torch.ones(pts.size(0), dtype=torch.bool)*segment
        pts = checkAffine(device, workingDirectory, pts, ptsSegmentation)
        pts = checkNonrigid(device, workingDirectory, pts, ptsSegmentation)

        p = igl.signed_distance(coords.cpu().numpy(), pts.cpu().numpy(), simplices.long().cpu().numpy())
        sdf = movingImage[0].clone().float().to(device=device)
        sdf[coords[:, 0].long(), coords[:, 1].long(), coords[:, 2].long()] = torch.from_numpy(p[0]).flatten().to(
            device=device)
        sdf = torch.where(sdf < 0, 1, 0)
        intersectionmap = intersectionmap + sdf

    #intersectionmap = intersectionmap[boundarySub:-boundarySub, boundarySub:-boundarySub, boundarySub:-boundarySub]
    intersectionmapZeros = intersectionmap.clone() * 0
    intersectionmapZeros[boundarySub:-boundarySub, boundarySub:-boundarySub, boundarySub:-boundarySub] = 1
    return intersectionmap*intersectionmapZeros

def measure_intersection2d(device,workingDirectory,voxelToMm=None,segmentsOfInterest=None):
    """ POSTPROCESSING OVERLAP-GAP
    Cdoes only intersection, no gap
    """

    #BASICS: load images
    movingImageMask=torch.unsqueeze(torch.from_numpy(np.load(os.path.join(workingDirectory, "moving_mask.npy"))), dim=0).to(device=device)
    movingImage=torch.unsqueeze(torch.from_numpy(np.load(os.path.join(workingDirectory, "moving.npy"))), dim=0).to(device=device)




    indices = np.indices(movingImage.cpu()[0].size())
    coords = np.empty((np.prod(movingImage.cpu().size()), len(movingImage[0].cpu().size())))
    for i, slide in enumerate(indices):
        coords[:, i] = slide.flatten()


    contours=[]
    for segment in segmentsOfInterest:
        maskTemp = torch.where(movingImageMask == segment, 1, 0)
        contour = torch.from_numpy(measure.find_contours(maskTemp[0].cpu().numpy(), 0.5)[0]).to(device=device)
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

        """print(f"Area of Intersection for contours {c[0]} and {c[1]}: {intersection_area}")
        fig, ax = plt.subplots()
        ax.imshow(movingImageMask.cpu()[0] * 0, cmap="binary")
        ax.plot(contours[c[0]].cpu().numpy()[:, 1], contours[c[0]].cpu().numpy()[:, 0], c='b', )
        ax.plot(contours[c[1]].cpu().numpy()[:, 1], contours[c[1]].cpu().numpy()[:, 0], c='r', )
        ax.set_xticks([])  # Remove x-axis ticks
        ax.set_yticks([])  # Remove y-axis ticks
        x, y = intersection.exterior.xy
        ax.fill(y, x, facecolor="green", edgecolor="green", alpha=1)
        ax.set_xlim([5, 25])  # Limit x-axis to range [30, 70]
        ax.set_ylim([2, 27])
        plt.savefig("/home/thomas/Pictures/synthetic_I/" + f'synthetic_I_ccdir.eps', format='eps', bbox_inches='tight')
        plt.close()
        #plt.show()"""
    return dictIntersection










