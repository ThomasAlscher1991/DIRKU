import pyvista as pv
from shapely.geometry import Polygon
from skimage import measure
import itertools
import torch
import numpy as np
import os
from .. import  geometricTransformations,  collisionDetection,utils, interpolation,meshing
import re
import matplotlib.pyplot as plt
import igl
import pickle
from .postprocessing_utils import *
from shapely.geometry import LineString, Polygon

from shapely.ops import polygonize, unary_union


def measure_selfIntersection3dVoxelBased(device,workingDirectory,voxelToMm=None,segmentsOfInterest=None):
    """ POSTPROCESSING SELF-INTERSECTION
    Calculates volumne/surface of self-intersections.
    Use the same interpolators, integrators, geometric transformations with the same class variables as used in the optimization.
    Set the following variables
        :param device: sets the computation device, see torch
        :type device: string
        :param workingDirectory: path to working directory, see docs
        :type workingDirectory: string
        :param voxelToMm: voxel or pixel size to mm; used to scale the image plot; one entry corresponding to each image dimension;
        :type voxelToMm: torch.tensor
    """
    # BASICS: load images
    movingImageMask = torch.unsqueeze(torch.from_numpy(np.load(os.path.join(workingDirectory, "moving_mask.npy"))),
                                      dim=0).to(device=device)

    indices = np.indices(movingImageMask.cpu()[0].size())
    coords = np.empty((np.prod(movingImageMask.cpu().size()), len(movingImageMask[0].cpu().size())))
    for i, slide in enumerate(indices):
        coords[:, i] = slide.flatten()
    coords = torch.from_numpy(coords).to(device=device).float()

    segmentName = str(segmentsOfInterest)

    mask=movingImageMask.clone()*0
    for s in segmentsOfInterest:
        mask=mask+torch.where(movingImageMask == s, 1, 0)

    m = meshing.surfaceMesh(mask[0], segmentName, device, workingDirectory, reuse=True)
    vertices, simplices = m.getVerticesAndSimplicesSurface()
    vertices = vertices.float()
    simplices = simplices.int()
    verticesSegmentation=utils.assignPoints(device, vertices, movingImageMask, segmentsOfInterest, initialValue=10000).long()

    vertices = checkAffine(device, workingDirectory, vertices, verticesSegmentation)
    vertices = checkNonrigid(device, workingDirectory, vertices, verticesSegmentation)

    verticesNumpy = vertices.cpu().numpy().copy(order='C')
    wn = igl.fast_winding_number_for_meshes(verticesNumpy, simplices.cpu().numpy(), coords.cpu().numpy())
    wn = np.round(wn, decimals=1)

    coordSel = coords.cpu().numpy()[wn > 1]

    volume = coordSel.shape[0] * np.prod(voxelToMm.cpu().numpy())
    print("volume of self intersection in cubic mm : ", str(volume))
    print("number of self intersection : ", coordSel.shape[0])

    return volume


def measure_selfIntersection3dVoxelBased_distanceToBoundaries(device,workingDirectory,voxelToMm=None,segmentsOfInterest=None):
    """ POSTPROCESSING SELF-INTERSECTION
    Calculates volumne/surface of self-intersections.
    Use the same interpolators, integrators, geometric transformations with the same class variables as used in the optimization.
    Set the following variables
        :param device: sets the computation device, see torch
        :type device: string
        :param workingDirectory: path to working directory, see docs
        :type workingDirectory: string
        :param voxelToMm: voxel or pixel size to mm; used to scale the image plot; one entry corresponding to each image dimension;
        :type voxelToMm: torch.tensor
    """
    # BASICS: load images
    movingImageMask = torch.unsqueeze(torch.from_numpy(np.load(os.path.join(workingDirectory, "moving_mask.npy"))),
                                      dim=0).to(device=device)

    indices = np.indices(movingImageMask.cpu()[0].size())
    coords = np.empty((np.prod(movingImageMask.cpu().size()), len(movingImageMask[0].cpu().size())))
    for i, slide in enumerate(indices):
        coords[:, i] = slide.flatten()
    coords = torch.from_numpy(coords).to(device=device).float()

    segmentName = str(segmentsOfInterest)

    mask=movingImageMask.clone()*0
    for s in segmentsOfInterest:
        mask=mask+torch.where(movingImageMask == s, 1, 0)

    m = meshing.surfaceMesh(mask[0], segmentName, device, workingDirectory, reuse=True)
    vertices, simplices = m.getVerticesAndSimplicesSurface()
    vertices = vertices.float()
    simplices = simplices.int()
    verticesSegmentation=utils.assignPoints(device, vertices, movingImageMask, segmentsOfInterest, initialValue=10000).long()

    vertices = checkAffine(device, workingDirectory, vertices, verticesSegmentation)
    vertices = checkNonrigid(device, workingDirectory, vertices, verticesSegmentation)

    verticesNumpy = vertices.cpu().numpy().copy(order='C')
    wn = igl.fast_winding_number_for_meshes(verticesNumpy, simplices.cpu().numpy(), coords.cpu().numpy())
    wn = np.round(wn, decimals=1)

    coordSel = coords.cpu().numpy()[wn > 1]
    if coordSel.shape[0]>0:

        coordSel=torch.from_numpy(coordSel).to(device=device)

        verticesFissure, faces = igl.read_triangle_mesh(
            os.path.join(workingDirectory, f"moving_fissure_{segmentsOfInterest}.stl"), 'float')

        verticesFissure=torch.from_numpy(verticesFissure).to(device=device)
        verticesFissureSegmentation=utils.assignPoints(device, verticesFissure, movingImageMask, segmentsOfInterest, initialValue=10000).long()

        verticesFissure = checkAffine(device, workingDirectory, verticesFissure.float(), verticesFissureSegmentation)
        verticesFissure = checkNonrigid(device, workingDirectory, verticesFissure, verticesFissureSegmentation)
        coordSel=coordSel*voxelToMm
        verticesFissure=verticesFissure*voxelToMm
        coordSelDist, i, c = igl.signed_distance(coordSel.cpu().numpy(), verticesFissure.cpu().numpy(), faces)
        coordSelDist=np.where(coordSelDist<0,0,coordSelDist)
        print("distance to fissure of intersections",np.mean(coordSelDist),np.std(coordSelDist))
        return coordSelDist
    else:
        return np.ones((10))*-1






def measure_selfIntersection3dPointBased(device,workingDirectory,voxelToMm=None,segmentsOfInterest=None):
    """ POSTPROCESSING SELF-INTERSECTION
    Calculates volumne/surface of self-intersections.
    Use the same interpolators, integrators, geometric transformations with the same class variables as used in the optimization.
    Set the following variables
        :param device: sets the computation device, see torch
        :type device: string
        :param workingDirectory: path to working directory, see docs
        :type workingDirectory: string
        :param voxelToMm: voxel or pixel size to mm; used to scale the image plot; one entry corresponding to each image dimension;
        :type voxelToMm: torch.tensor
    """
    movingImageMask = torch.unsqueeze(torch.from_numpy(np.load(os.path.join(workingDirectory, "moving_mask.npy"))),
                                      dim=0).to(device=device)

    indices = np.indices(movingImageMask.cpu()[0].size())


    segmentName = str(segmentsOfInterest)

    mask = movingImageMask.clone() * 0
    for s in segmentsOfInterest:
        mask = mask + torch.where(movingImageMask == s, 1, 0)

    m = meshing.surfaceMesh(mask[0], segmentName, device, workingDirectory, reuse=True)
    vertices, simplices = m.getVerticesAndSimplicesSurface()
    vertices = vertices.float()
    verticesOrig=vertices.clone()
    simplices = simplices.int()
    verticesSegmentation = utils.assignPoints(device, vertices, mask, movingImageMask, initialValue=10000)

    vertices = checkAffine(device, workingDirectory, vertices, verticesSegmentation)
    vertices = checkNonrigid(device, workingDirectory, vertices, verticesSegmentation)


    intensityInterpolator = interpolation.cubic(device, torch.tensor([1., 1., 1.], device=device))

    m = meshing.surfaceMesh(mask[0], segmentName, device, workingDirectory, reuse=True)
    m.getVerticesAndSimplicesSurface()
    s=utils.sdfCreator(device,workingDirectory=workingDirectory,reuse=True)
    sdf=s.fromMesh(os.path.join(workingDirectory, "reuse", f"surface_mesh_segment_{segmentName}.stl"),mask)
    grads=s.getGrads(sdf)

    d=collisionDetection.selfintersectionDetection(verticesOrig,sdf,intensityInterpolator,grads,simplices,verticesOrig,1,device)
    depth, intersectingVertices,selector,intersectingSimplices=d.intersectionsNodes(vertices)

    numberOfIntersections=intersectingVertices.size(0)
    print(f"number of self intersections: ", numberOfIntersections)
    print(" sum of depth: ",depth)


    return numberOfIntersections,depth




def measure_selfIntersection2dContourBased(device,workingDirectory,voxelToMm=None,segmentsOfInterest=None):
    """ POSTPROCESSING SELF-INTERSECTION
    both measure and voisual because of how shapely detects selfintersections

    """

    #BASICS: load images
    movingImageMask=torch.unsqueeze(torch.from_numpy(np.load(os.path.join(workingDirectory, "moving_mask.npy"))), dim=0).to(device=device)


    mask=movingImageMask.clone()*0
    for s in segmentsOfInterest:
        mask=mask+torch.where(movingImageMask==s,1,0)


    contour = torch.from_numpy(measure.find_contours((mask[0]).cpu().numpy(), 0.6)[0]).to(device=device)
    inter = interpolation.nearest(device, scale=torch.tensor([1., 1.], device=device))
    segmentation = inter(contour, movingImageMask)

    contour = checkAffine(device, workingDirectory, contour, segmentation)
    contour = checkNonrigid(device, workingDirectory, contour, segmentation)

    line1 = LineString(contour.cpu().numpy())
    if line1.is_closed:
        polygon1 = Polygon(line1)
    else:
        raise Exception("Postprocessing contour not closed")
    lines = LineString(polygon1.exterior.coords)
    intersections = unary_union(
        [lines.intersection(LineString([lines.coords[i], lines.coords[i + 1]])) for i in range(len(lines.coords) - 1)])
    intersecting_areas = list(polygonize(intersections))
    sum = 0
    i = 0
    listIntersection=[]
    for poly in intersecting_areas:
        listIntersection.append(poly.area)
        sum = sum + poly.area
        i = i + 1
        print(i, poly.area)

        fig, ax = plt.subplots()
        ax.imshow(movingImageMask.cpu()[0] * 0, cmap="binary")
        ax.plot(contour[:, 1].cpu(), contour[:, 0].cpu(), c='r', )
        ax.set_xticks([])  # Remove x-axis ticks
        ax.set_yticks([])  # Remove y-axis ticks
        x, y = poly.exterior.xy
        ax.fill(y, x, facecolor="green", edgecolor="green", alpha=1)
        ax.set_xlim([10, 90])  # Limit x-axis to range [30, 70]
        ax.set_ylim([20, 80])
        plt.savefig("/home/thomas/Pictures/synthetic_SI/" + f'synthetic_fixed_SI_{i}_ccdir.eps', format='eps',
                    bbox_inches='tight')
        plt.close()

    print(f"The area of the self-intersecting region is: {sum}")
    return listIntersection






