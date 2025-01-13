from skimage import measure
from .. import  geometricTransformations,  collisionDetection,utils, interpolation
import matplotlib.pyplot as plt
import igl
from .postprocessing_utils import *
from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize, unary_union
from typing import Optional, Type, Union, Tuple
from torch import Tensor
import skimage
def measure_selfIntersection3d(device: str,workingDirectory: str,voxelToMm: Optional[Tensor]=None,segmentsOfInterest: Optional[Tensor]=None,vertices: Tensor=None,simplices: Tensor=None)->Tensor:
    """ Calculates the overlap from selfintersections in 3D. Object is input as vertex and simplex collection.
        :param device: sets the computation device, see torch
        :type device: string
        :param workingDirectory: path to working directory, see docs
        :type workingDirectory: string
        :param segmentsOfInterest: segmentations of interest list
        :type segmentsOfInterest: list
        :param voxelToMm: cell dimensions in mm
        :type voxelToMm: torch.Tensor
        :param vertices: vertex collection
        :type vertices: list
        :param simplices: simplex collection
        :type simplices: list
        :return : overlap
        :rtype : Tensor
    """
    # BASICS: load images
    movingImageMask = torch.unsqueeze(torch.from_numpy(np.load(os.path.join(workingDirectory, "moving_mask.npy"))),
                                      dim=0).to(device=device)

    indices = np.indices(movingImageMask.cpu()[0].size())
    coords = np.empty((np.prod(movingImageMask.cpu().size()), len(movingImageMask[0].cpu().size())))
    for i, slide in enumerate(indices):
        coords[:, i] = slide.flatten()
    coords = torch.from_numpy(coords).to(device=device).float()

    verticesSegmentation=utils.assignPoints(device, vertices, movingImageMask, segmentsOfInterest, initialValue=10000).long()

    vertices = checkAffine(device, workingDirectory, vertices, verticesSegmentation)
    vertices = checkNonrigid(device, workingDirectory, vertices, verticesSegmentation)

    verticesNumpy = vertices.cpu().numpy().copy(order='C')
    wn = igl.fast_winding_number_for_meshes(verticesNumpy, simplices.cpu().numpy(), coords.cpu().numpy())
    wn = np.round(wn, decimals=1)

    coordSel = coords.cpu().numpy()[wn > 1]

    if voxelToMm is not None:
        volume = coordSel.shape[0] * np.prod(voxelToMm.cpu().numpy())
    else:
        volume = coordSel.shape[0]
    return volume


def measure_selfIntersection2d(device: str,workingDirectory: str,voxelToMm: Optional[Tensor]=None,segmentsOfInterest: Optional[list]=None)->Tensor:
    """ Calculates the overlap from selfintersections in 2D. Object is input as contour.
        :param device: sets the computation device, see torch
        :type device: string
        :param workingDirectory: path to working directory, see docs
        :type workingDirectory: string
        :param segmentsOfInterest: segmentations of interest list
        :type segmentsOfInterest: list
        :param voxelToMm: cell dimensions in mm
        :type voxelToMm: torch.Tensor
        :return : overlap
        :rtype : Tensor
    """
    movingImageMask = torch.unsqueeze(torch.from_numpy(np.load(os.path.join(workingDirectory, "moving_mask.npy"))),
                                      dim=0).to(device=device)
    inter = interpolation.nearest(device, scale=torch.tensor([1., 1.], device=device))
    movingImageMaskCont = torch.from_numpy(np.load(os.path.join(workingDirectory, "moving_mask.npy"))).to(device=device)
    mask=torch.zeros(movingImageMaskCont.size(),device=device)

    for segment in segmentsOfInterest:
        maskTemp=torch.where(movingImageMaskCont==segment,1,0).to(device=device)
        mask=mask+maskTemp
    print(mask.size())

    contour=skimage.measure.find_contours(mask.cpu().numpy(),level=0.5)[0]
    contour=torch.from_numpy(contour).to(device=device)


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
        plt.savefig(workingDirectory + f'/results/synthetic_fixed_SI_{i}_ccdir.eps', format='eps',
                    bbox_inches='tight')
        plt.close()

    print(f"The area of the self-intersecting region is: {sum}")
    return listIntersection






