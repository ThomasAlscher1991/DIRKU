import torch
import numpy as np
from scipy import ndimage
import os
from .. import  geometricTransformations
import pickle
from .postprocessing_utils import *
import math

def round_to_x_decimals_first_nonzero(value,x):
    order_of_magnitude = torch.floor(torch.log10(torch.abs(value)))
    scaled_value = value / (10 ** order_of_magnitude)
    rounded_scaled_value = torch.round(scaled_value, decimals=x)
    return rounded_scaled_value * (10 ** order_of_magnitude)

def negJacobian(device,workingDirectory,voxelToMm=None):
    """ POSTPROCESSING NEGATIVE JACOBIANS.
    Sample script for calculating the percentage of determinant of negative Jacobians. Uses finite differences.
    Use the same interpolators, integrators, geometric transformations with the same class variables as used in the optimization.
    For interpolation of mask, use either nearest neighbour interpolation or round result to integers.
    Set the following variables
        :param device: sets the computation device, see torch
        :type device: string
        :param workingDirectory: path to working directory, see docs
        :type workingDirectory: string
        :param voxelToMm: voxel or pixel size to mm; used to scale the image plot; one entry corresponding to each image dimension;
        :type voxelToMm: torch.tensor
    """

    #BASICS: load images
    movingImageMask=torch.unsqueeze(torch.from_numpy(np.load(os.path.join(workingDirectory, "moving_mask.npy"))), dim=0).to(device=device)
    fixedImageMask=torch.unsqueeze(torch.from_numpy(np.load(os.path.join(workingDirectory, "fixed_mask.npy"))), dim=0).to(device=device)
    indices = np.indices(movingImageMask.cpu()[0].size())
    pts = np.empty((np.prod(movingImageMask.cpu().size()), len(movingImageMask[0].cpu().size())))
    for i, slide in enumerate(indices):
        pts[:, i] = slide.flatten()
    pts=torch.from_numpy(pts).to(device=device).float()




    # Apply affine
    if os.path.exists(os.path.join(workingDirectory, "results", "transformation_affine.npy")):
        print("general affine")

        affineMat = torch.from_numpy(
            np.load(os.path.join(workingDirectory, "results", "transformation_affine.npy"))).to(device=device)
        affine = geometricTransformations.affineTransformation(pts)
        pts = affine.apply(pts, affineMat)
    else:
        pass
    listOfSegments = torch.unique(movingImageMask)
    for segment in listOfSegments:
        if os.path.exists(os.path.join(workingDirectory, "results", f"transformation_affine_{segment}.npy")):
            print(f"segement affine {segment}")

            affineMat = torch.from_numpy(
                np.load(os.path.join(workingDirectory, "results", f"transformation_affine_{segment}.npy"))).to(
                device=device)
            ptsSegment = pts[movingImageMask.flatten() == segment]
            affine = geometricTransformations.affineTransformation(ptsSegment)
            ptsSegment = affine.apply(ptsSegment, affineMat)
            pts[movingImageMask.flatten() == segment]= ptsSegment
        else:
            pass

    # Get solutions
    files = os.listdir(os.path.join(workingDirectory, "results"))
    filtered_files = [file for file in files if file.startswith("transformation_nonrigid") and file.endswith(".npy")]

    # Apply nonrigid
    segmentsList = []

    if len(filtered_files) > 0:
        sorted_files = sorted(filtered_files,
                              key=lambda s: (extract_segment_and_scale_NPY(s)[0], extract_segment_and_scale_NPY(s)[1]),
                              reverse=True)
        for f in sorted_files:
            velocityField = torch.from_numpy(np.load(os.path.join(workingDirectory, "results", f))).to(device=device)
            scale,segment=extract_segment_and_scale_NPY(f)
            if segment is None:
                print("general deformable registreiton")

                scale = torch.tensor(scale).to(device=device)
                with open(os.path.join(workingDirectory, "results",
                                       "class_transformation_nonrigid_scale_" + str(scale.cpu()) + ".pkl"),
                          'rb') as input_file:
                    nrDeformation = pickle.load(input_file)
                pts = nrDeformation.apply(pts, velocityField)

            else:
                print(f"segment deformable registreiton {segment}")

                ptsSegment = pts[movingImageMask.flatten() == segment]
                scale = torch.tensor(scale).to(device=device)
                with open(os.path.join(workingDirectory, "results",
                                       f"class_transformation_nonrigid_segment_{segment}_scale_" + str(
                                           scale.cpu()) + ".pkl"), 'rb') as input_file:
                    nrDeformation = pickle.load(input_file)
                ptsSegment = nrDeformation.apply(ptsSegment, velocityField)
                pts[movingImageMask.flatten() == segment] = ptsSegment
                if segment not in segmentsList:
                    segmentsList.append(segment)


    if pts.size(1)==3:
        #print("% NEGATIVE DET(JAC)")
        phiX=(pts[:,0]).flatten().reshape(fixedImageMask.size())[0]*voxelToMm[0]
        phiY=(pts[:,1]).flatten().reshape(fixedImageMask.size())[0]*voxelToMm[1]
        phiZ=(pts[:,2]).flatten().reshape(fixedImageMask.size())[0]*voxelToMm[2]

        xphiX,yphiX,zphiX=compute_gradient_central_diff3D(phiX)
        xphiY,yphiY,zphiY=compute_gradient_central_diff3D(phiY)
        xphiZ,yphiZ,zphiZ=compute_gradient_central_diff3D(phiZ)

        id = torch.eye(3)
        id = id.reshape((1, 3, 3))
        id = id.repeat(phiX.size(0), phiX.size(1), phiX.size(2), 1, 1)

        jacPhi=torch.zeros(id.size())
        jacPhi[:,:,:,0,0]=xphiX
        jacPhi[:,:,:,0,1]=yphiX
        jacPhi[:,:,:,0,2]=zphiX
        jacPhi[:,:,:,1,0]=xphiY
        jacPhi[:,:,:,1,1]=yphiY
        jacPhi[:,:,:,1,2]=zphiY
        jacPhi[:,:,:,2,0]=xphiZ
        jacPhi[:,:,:,2,1]=yphiZ
        jacPhi[:,:,:,2,2]=zphiZ
        detJacPhi=torch.linalg.det(jacPhi)

        if torch.min(detJacPhi)>=0:
            #print("no negative det(jac)")
            pass


        neg=torch.where(detJacPhi<0,1,0)
        procent=torch.sum(neg)/torch.numel(neg)
        procentOver=procent*100

        #print("|% of negative Jacobian elements overall| ", procent*100,"|")

        #get boundaries
        boundaries=torch.zeros(movingImageMask.size())[0]
        boundaries=boundaries.numpy()

        for segment in segmentsList:
            temp=movingImageMask.clone().cpu().numpy()[0]
            temp=np.where(temp==segment,1,0)
            temp_ero=ndimage.binary_erosion(temp)
            temp_bound=temp-temp_ero
            boundaries=boundaries+temp_bound
        boundaries[0,:,:]=0
        boundaries[-1,:,:]=0
        boundaries[:,0,:]=0
        boundaries[:,-1,:]=0
        boundaries[:,:,0]=0
        boundaries[:,:,-1]=0

        boundaries = np.where(boundaries > 0, 1, 0)
        boundaries = torch.from_numpy(boundaries)

        detJacPhiNoBoundaries = detJacPhi.flatten()[boundaries.flatten() == 0]



        if torch.min(detJacPhiNoBoundaries)>=0:
            #print("no negative det(jac) excluding boundaries")
            pass


        neg=torch.where(detJacPhiNoBoundaries<0,1,0)
        procent=torch.sum(neg)/torch.numel(neg)
        procentOver=procent*100
        procent=torch.sum(neg)/torch.numel(neg)

        print("|% of negative Jacobian elements excluding segment boundaries| ", procent*100,"|")

        return detJacPhi,detJacPhiNoBoundaries
    else:
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

        detJacPhi = torch.linalg.det(jacPhi)

        neg = torch.where(detJacPhi < 0, 1, 0)
        procent = torch.sum(neg) / torch.numel(neg)

        print("|% of negative Jacobian elements overall| ", procent*100,"|")

        # get boundaries
        boundaries = torch.zeros(movingImageMask.size())[0]
        boundaries = boundaries.numpy()

        for segment in segmentsList:
            temp = movingImageMask.clone().cpu().numpy()[0]
            temp = np.where(temp == segment, 1, 0)
            temp_ero = ndimage.binary_erosion(temp)
            temp_bound = temp - temp_ero
            boundaries = boundaries + temp_bound
        boundaries[0, :] = 0
        boundaries[-1, :] = 0
        boundaries[:, 0] = 0
        boundaries[:, -1] = 0


        boundaries = np.where(boundaries > 0, 1, 0)
        boundaries = torch.from_numpy(boundaries)

        negNoBoundaries = neg.flatten()[boundaries.flatten() == 0]

        procent = torch.sum(negNoBoundaries) / torch.numel(negNoBoundaries)

        print("|% of negative Jacobian elements excluding segment boundaries| ", procent*100,"|")












