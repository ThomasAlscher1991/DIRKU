import torch
import re
import numpy as np
import os
import pickle
from ..import geometricTransformations
from typing import Optional, Type, Union, Tuple
from torch import Tensor

def compute_gradient_central_diff3D(F: Tensor, dx: Optional[float]=1.0, dy: Optional[float]=1.0, dz: Optional[float]=1.0)->Tuple[Tensor,Tensor,Tensor]:
    """ Computes Jacobian of dense vector fields via finite differences in 3D.
    Set the following variables
        :param F: field
        :type F: torch.Tensor
        :param dx: finite differences step in x direction
        :type dx: float
        :param dy: finite differences step in x direction
        :type dy: float
        :param dz: finite differences step in x direction
        :type dz: float
        :return dFdx: derivative in x
        :rtype dFdx: torch.Tensor
        :return dFdy: derivative in y
        :rtype dFdy: torch.Tensor
        :return dFdz: derivative in z
        :rtype dFdz: torch.Tensor
    """
    dFdx = (torch.roll(F, -1, dims=0) - torch.roll(F, 1, dims=0)) / (2 * dx)
    dFdy = (torch.roll(F, -1, dims=1) - torch.roll(F, 1, dims=1)) / (2 * dy)
    dFdz = (torch.roll(F, -1, dims=2) - torch.roll(F, 1, dims=2)) / (2 * dz)
    dFdx[0, :, :] = (F[1, :, :] - F[0, :, :]) / dx  # Forward difference
    dFdx[-1, :, :] = (F[-1, :, :] - F[-2, :, :]) / dx  # Backward difference
    dFdy[:, 0, :] = (F[:, 1, :] - F[:, 0, :]) / dy  # Forward difference
    dFdy[:, -1, :] = (F[:, -1, :] - F[:, -2, :]) / dy  # Backward difference
    dFdz[:, :, 0] = (F[:, :, 1] - F[:, :, 0]) / dz  # Forward difference
    dFdz[:, :, -1] = (F[:, :, -1] - F[:, :, -2]) / dz  # Backward difference
    return dFdx, dFdy, dFdz

def compute_gradient_central_diff2D(F: Tensor, dx: Optional[float]=1.0, dy: Optional[float]=1.0)->Tuple[Tensor,Tensor]:
    """ Computes jacobian via finite differences in 2D.
    Set the following variables
        :param F: field
        :type F: torch.Tensor
        :param dx: finite differences step in x direction
        :type dx: float
        :param dy: finite differences step in x direction
        :type dy: float
        :return dFdx: derivative in x
        :rtype dFdx: torch.tensor
        :return dFdy: derivative in y
        :rtype dFdy: torch.Tensor
    """
    dFdx = (torch.roll(F, -1, dims=0) - torch.roll(F, 1, dims=0)) / (2 * dx)
    dFdy = (torch.roll(F, -1, dims=1) - torch.roll(F, 1, dims=1)) / (2 * dy)
    dFdx[0, :] = (F[1, :] - F[0, :]) / dx  # Forward difference
    dFdx[-1, :] = (F[-1, :] - F[-2, :]) / dx  # Backward difference
    dFdy[:, 0] = (F[:, 1] - F[:, 0]) / dy  # Forward difference
    dFdy[:, -1] = (F[:, -1] - F[:, -2]) / dy  # Backward difference
    return dFdx, dFdy

def extract_segment_and_scale_JSON(file: str)->Tuple[str,str]:
    """ For automated recognition of scale and segment in nonrigid result names for json files.
        Handles single integer segments, tuple segments, and 'None' segments
        :param file: file name of result
        :type file: str
        :return scale_values: segment
        :rtype scale_values: str
        :return segment_values: scale
        :rtype segment_values: str
    """
    # Updated pattern to handle single integers, tuples, and the word 'None'
    pattern = r"transformation_nonrigid_segment_(?P<segment>\(\d+(?:, \d+)*\)|\d+|None)_scale_tensor\(\[(?P<scale>[0-9., ]+)\]\)\.json"
    # Loop over file names and extract information
    match = re.match(pattern, file)
    if match:
        segment = match.group("segment")
        scale = match.group("scale")
        # Parse the segment: handle tuple, single integer, and 'None' case
        if segment == 'None':
            segment_values = None
        elif segment.startswith("(") and segment.endswith(")"):
            # Parse tuple segment
            segment_values = tuple(int(val.strip()) for val in segment[1:-1].split(','))
        else:
            # Parse single integer segment
            segment_values = int(segment)
        # Parse the scale as a list of floats
        scale_values = [float(val.strip()) for val in scale.split(',')]

        return scale_values, segment_values
    else:
        return None, None

def extract_segment_and_scale_NPY(file: str)->Tuple[str,str]:
    """ For automated recognition of scale and segment in nonrigid result names for npy files.
        Handles single integer segments, tuple segments, and 'None' segments
        :param file: file name of result
        :type file: str
        :return scale_values: segment
        :rtype scale_values: str
        :return segment_values: scale
        :rtype segment_values: str
    """
    # Updated pattern to handle single integers, tuples, and the word 'None'
    pattern = r"transformation_nonrigid_segment_(?P<segment>\(\d+(?:, \d+)*\)|\d+|None)_scale_tensor\(\[(?P<scale>[0-9., ]+)\]\)\.npy"

    # Loop over file names and extract information
    match = re.match(pattern, file)
    if match:
        segment = match.group("segment")
        scale = match.group("scale")

        # Parse the segment: handle tuple, single integer, and 'None' case
        if segment == 'None':
            segment_values = None
        elif segment.startswith("(") and segment.endswith(")"):
            # Parse tuple segment
            segment_values = tuple(int(val.strip()) for val in segment[1:-1].split(','))
        else:
            # Parse single integer segment
            segment_values = int(segment)

        # Parse the scale as a list of floats
        scale_values = [float(val.strip()) for val in scale.split(',')]

        return scale_values, segment_values
    else:
        return None, None
def extract_scale_NPY(file: str)->str:
    """ For automated recognition of scale in nonrigid result names for npy files.
        :param file: file name of result
        :type file: str
        :return scale_values: segment
        :rtype scale_values: str
    """
    # Updated pattern to handle single integers, tuples, and the word 'None'
    pattern = r"transformation_nonrigid_scale_tensor\(\[(?P<scale>[0-9., ]+)\]\)\.npy"

    # Loop over file names and extract information
    match = re.match(pattern, file)
    if match:
        scale = match.group("scale")
        scale_values = [float(val.strip()) for val in scale.split(',')]

        return scale_values
    else:
        return None
def checkAffine(device: str,workingDirectory: str,pts: Tensor,segmentations: Optional[list]=None)->Tensor:
    """ Checks the results folder for affine transformations and applies them.
         :param device: file name of result
         :type device: str
         :param workingDirectory: workingDirectory path
         :type workingDirectory: str
         :param pts: points to be transformed
         :type pts: torch.Tensor
         :param segmentations: segmentations of interest list
         :type segmentations: list
         :return pts: transformed points
         :rtype pts: torch.Tensor
     """
    if segmentations is not None:
        for segment in torch.unique(segmentations):
            if os.path.exists(os.path.join(workingDirectory, "results", f"transformation_affine_{segment}.npy")):
                print(f" segment {segment} affine registration applied ")
                affineMat = torch.from_numpy(
                    np.load(os.path.join(workingDirectory, "results", f"transformation_affine_{segment}.npy"))).to(
                    device=device)
                ptsSegment = pts[segmentations.flatten() == segment]
                affine = geometricTransformations.affineTransformation(ptsSegment)
                ptsSegment = affine.apply(ptsSegment, affineMat)
                pts[segmentations.flatten() == segment] = ptsSegment
    else:
        if os.path.exists(os.path.join(workingDirectory, "results", "transformation_affine.npy")):
            print("overall affine registration applied")
            affineMat = torch.from_numpy(
                np.load(os.path.join(workingDirectory, "results", "transformation_affine.npy"))).to(device=device)
            affine = geometricTransformations.affineTransformation(pts)
            pts = affine.apply(pts, affineMat)
    return pts

def checkNonrigid(device: str,workingDirectory: str,pts: Tensor,segmentations: Optional[list]=None)-> Tensor:
    """ Checks the results folder for nonrigied transformations and applies them.
         :param device: file name of result
         :type device: str
         :param workingDirectory: workingDirectory path
         :type workingDirectory: str
         :param pts: points to be transformed
         :type pts: torch.Tensor
         :param segmentations: segmentations of interest list
         :type segmentations: list
         :return pts: transformed points
         :rtype pts: torch.Tensor
     """
    if segmentations is not None:
        files = os.listdir(os.path.join(workingDirectory, "results"))
        filtered_files = [file for file in files if
                          file.startswith("transformation_nonrigid_segment") and file.endswith(".npy")]
        sorted_files = sorted(filtered_files,
                              key=lambda s: (extract_segment_and_scale_NPY(s)[0], extract_segment_and_scale_NPY(s)[1]),
                              reverse=True)
        for f in sorted_files:
            velocityField = torch.from_numpy(np.load(os.path.join(workingDirectory, "results", f))).to(device=device)
            scale,segment=extract_segment_and_scale_NPY(f)


            ptsSegment = pts[segmentations.flatten() == segment]

            if ptsSegment.size(0)>0:
                pass
                print(f" segment {segment} deformable registration applied at scale {scale}")

            scale = torch.tensor(scale).to(device=device)
            with open(os.path.join(workingDirectory, "results",
                                   f"class_transformation_nonrigid_segment_{segment}_scale_" + str(
                                       scale.cpu()) + ".pkl"), 'rb') as input_file:
                nrDeformation = pickle.load(input_file)
            ptsSegment = nrDeformation.apply(ptsSegment, velocityField)
            pts[segmentations.flatten() == segment] = ptsSegment

    else:
        files = os.listdir(os.path.join(workingDirectory, "results"))
        filtered_files = [file for file in files if
                          file.startswith("transformation_nonrigid") and file.endswith(".npy")]
        sorted_files = sorted(filtered_files,
                              key=lambda s: (extract_scale_NPY(s)[0]),
                              reverse=True)
        for f in sorted_files:
            velocityField = torch.from_numpy(np.load(os.path.join(workingDirectory, "results", f))).to(device=device)
            scale=extract_scale_NPY(f)
            print(f" general deformable registration applied at scale {scale}")

            scale = torch.tensor(scale).to(device=device)
            with open(os.path.join(workingDirectory, "results",
                                   "class_transformation_nonrigid_scale_" + str(scale.cpu()) + ".pkl"),
                      'rb') as input_file:
                nrDeformation = pickle.load(input_file)
            pts = nrDeformation.apply(pts, velocityField)
    return pts

def checkAffineInverse(device: str,workingDirectory: str,pts: Tensor,segmentations: Optional[list]=None)->Tensor:
    """ Checks the results folder for affine transformations and applies the inverse.
         :param device: file name of result
         :type device: str
         :param workingDirectory: workingDirectory path
         :type workingDirectory: str
         :param pts: points to be transformed
         :type pts: torch.Tensor
         :param segmentations: segmentations of interest list
         :type segmentations: list
         :return pts: transformed points
         :rtype pts: torch.Tensor
     """
    if segmentations is not None:
        for segment in torch.unique(segmentations):
            if os.path.exists(os.path.join(workingDirectory, "results", f"transformation_affine_{segment}.npy")):
                #print(f" segment {segment} inverse affine registration applied ")
                affineMat = torch.from_numpy(
                    np.load(os.path.join(workingDirectory, "results", f"transformation_affine_{segment}.npy"))).to(
                    device=device)
                affineMat=torch.linalg.inv(affineMat)
                ptsSegment = pts[segmentations.flatten() == segment]
                affine = geometricTransformations.affineTransformation(ptsSegment)
                ptsSegment = affine.apply(ptsSegment, affineMat)
                pts[segmentations.flatten() == segment] = ptsSegment
    else:
        if os.path.exists(os.path.join(workingDirectory, "results", "transformation_affine.npy")):
            print("overall inverse affine registration applied")
            affineMat = torch.from_numpy(
                np.load(os.path.join(workingDirectory, "results", "transformation_affine.npy"))).to(device=device)
            affineMat = torch.linalg.inv(affineMat)

            affine = geometricTransformations.affineTransformation(pts)
            pts = affine.apply(pts, affineMat)
    return pts

def checkNonrigidInverse(device: str,workingDirectory: str,pts: Tensor,segmentations: Optional[list]=None)-> Tensor:
    """ Checks the results folder for nonrigied transformations and applies the inverse.
         :param device: file name of result
         :type device: str
         :param workingDirectory: workingDirectory path
         :type workingDirectory: str
         :param pts: points to be transformed
         :type pts: torch.Tensor
         :param segmentations: segmentations of interest list
         :type segmentations: list
         :return pts: transformed points
         :rtype pts: torch.Tensor
     """
    if segmentations is not None:
        files = os.listdir(os.path.join(workingDirectory, "results"))
        filtered_files = [file for file in files if
                          file.startswith("transformation_nonrigid_segment") and file.endswith(".npy")]
        sorted_files = sorted(filtered_files,
                              key=lambda s: (extract_segment_and_scale_NPY(s)[0], extract_segment_and_scale_NPY(s)[1]),
                              reverse=False)
        for f in sorted_files:
            velocityField = torch.from_numpy(np.load(os.path.join(workingDirectory, "results", f))).to(device=device)
            velocityField = torch.flip(velocityField, [0]) * -1
            scale,segment=extract_segment_and_scale_NPY(f)


            ptsSegment = pts[segmentations.flatten() == segment]
            if ptsSegment.size(0)>0:
                pass
                #print(f" segment {segment} deformable registration applied at scale {scale}")
            scale = torch.tensor(scale).to(device=device)
            with open(os.path.join(workingDirectory, "results",
                                   f"class_transformation_nonrigid_segment_{segment}_scale_" + str(
                                       scale.cpu()) + ".pkl"), 'rb') as input_file:
                nrDeformation = pickle.load(input_file)
            ptsSegment = nrDeformation.apply(ptsSegment, velocityField)
            pts[segmentations.flatten() == segment] = ptsSegment

    else:
        files = os.listdir(os.path.join(workingDirectory, "results"))
        filtered_files = [file for file in files if
                          file.startswith("transformation_nonrigid") and file.endswith(".npy")]
        sorted_files = sorted(filtered_files,
                              key=lambda s: (extract_scale_NPY(s)[0]),
                              reverse=False)
        for f in sorted_files:
            velocityField = torch.from_numpy(np.load(os.path.join(workingDirectory, "results", f))).to(device=device)
            velocityField = torch.flip(velocityField, [0]) * -1
            scale=extract_scale_NPY(f)
            print(f" general deformable registration applied at scale {scale}")

            scale = torch.tensor(scale).to(device=device)
            with open(os.path.join(workingDirectory, "results",
                                   "class_transformation_nonrigid_scale_" + str(scale.cpu()) + ".pkl"),
                      'rb') as input_file:
                nrDeformation = pickle.load(input_file)
            pts = nrDeformation.apply(pts, velocityField)
    return pts
