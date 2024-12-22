
import matplotlib.pyplot as plt
import torch
import numpy as np
import os, glob, tqdm
import json
from src.dirku import utils, interpolation, geometricTransformations, similarityMeasure, optimization, \
    numericalIntegration, collisionDetection, regularization,postprocessing
import pickle
import matplotlib.tri as mtri
import scipy
import skfmm
import sys
import time


def deformable_ccdir(device, workingDirectory, segmentTuple,simCoef=1,lr=1,colCoef=0):
    """DEMO DEFORMABLE PIECEWISE ADMM
    This sample script registers the moving image to the fixed image in 3D by computing independent registrations for each segment. Uses ADMM to alterantively optimize 2 cost terms.
    It first applies an affine registration for the whole domain,
    followed by multi-scale deformable registrations for every single segment.
    Saves the parameters and independent variables.
            :param device: sets the computation device, see torch
            :type device: string
            :param workingDirectory: path to working directory, see docs
            :type workingDirectory: string
            :param **kwargs: keyworded arguments; can be used to automate and vary experiments with different parameters
            :type **kwargs: any
    """
    # BASICS: clean result folder

    # BASICS: load images
    movingImage = torch.unsqueeze(torch.from_numpy(np.load(os.path.join(workingDirectory, "moving.npy"))),
                                  dim=0).to(device=device)
    fixedImage = torch.unsqueeze(torch.from_numpy(np.load(os.path.join(workingDirectory, "fixed.npy"))),
                                 dim=0).to(device=device)
    movingImageMask = torch.unsqueeze(
        torch.from_numpy(np.load(os.path.join(workingDirectory, "moving_mask.npy"))), dim=0).to(device=device)

    movingImageMaskTemp=movingImageMask.clone()*0
    for s in segmentTuple:
        movingImageMaskTemp=movingImageMaskTemp+torch.where(movingImageMask==s,1,0)


    """contour = torch.from_numpy(np.load(os.path.join(workingDirectory, "contours.npy"))).to(device=device)
    contour = torch.unique(contour, dim=0)
    m = meshing.triangleMesh(device)
    m.getTriangleMeshfromContour(contour)
    vertices, simp = m.getVerticesAndSimplices()
    m.cleanConvex(simp.cpu().numpy(), vertices.cpu().numpy(), torch.where(movingImageMask[0] > 0, 1, 0).cpu().numpy())
    vert, simp = m.getVerticesAndSimplices()
    selfColSel2 = vert[:, 0] <= 50.5
    selfColSel1 = vert[:, 0] > 50.5"""
    s = utils.sdfCreator(device, reuse=False)
    sdf = s.fromMask(torch.where(movingImageMask > 0, 1, 0), invert=True)

    vertices=torch.from_numpy(np.load(os.path.join(workingDirectory,"vertices.npy"))).to(device=device)
    simplices=torch.from_numpy(np.load(os.path.join(workingDirectory,"simplices.npy"))).to(device=device)
    shared1=torch.from_numpy(np.load(os.path.join(workingDirectory,"connectorPoints.npy"))).to(device=device)
    shared2=torch.from_numpy(np.load(os.path.join(workingDirectory,"connectorPoints.npy"))).to(device=device)

    verticesOriginal=vertices.clone()


    evalPointsSegment1 = utils.getEvaluationPoints(device,movingImageMask,  mask=movingImageMask, maskLabel=torch.tensor([segmentTuple[0]], device=device))
    evalPointsSegment2 = utils.getEvaluationPoints(device,movingImageMask,  mask=movingImageMask, maskLabel=torch.tensor([segmentTuple[1]], device=device))


    intensityInterpolator = interpolation.cubic(device, torch.tensor([1., 1.], device=device))
    evalPointsIntensitySegment1, _, _ = intensityInterpolator(evalPointsSegment1, movingImage)
    evalPointsIntensitySegment2, _, _ = intensityInterpolator(evalPointsSegment2, movingImage)
    scales = torch.tensor([[10., 10.],[5.,5.]], device=device)


    for scale in scales:
        tqdm.tqdm.write(f"Scale {scale}")
        decisionVariablesSegment1 = utils.getGridPoints(movingImage, scale, 1)
        decisionVariablesSegment1.requires_grad = True
        decisionVariablesSegment2 = utils.getGridPoints(movingImage, scale, 1)
        decisionVariablesSegment2.requires_grad = True
        vfInterpolator = interpolation.cubic(device, scale)

        integrator = numericalIntegration.forwardEuler(1, stationary=False)


        points1 = torch.concatenate((evalPointsSegment1.to(device=device), vertices))
        pointMask1 = torch.zeros((points1.size(0)), device=device)
        pointMask1[:evalPointsSegment1.size(0)] = 1
        points2 = torch.concatenate((evalPointsSegment2.to(device=device), vertices))
        pointMask2 = torch.zeros((points2.size(0)), device=device)
        pointMask2[:evalPointsSegment2.size(0)] = 1



        nrDeformationSegment1 = geometricTransformations.nonrigidDeformation(points1, integrator,
                                                                      vfInterpolator,pointMask1)
        nrDeformationSegment2 = geometricTransformations.nonrigidDeformation(points2, integrator,
                                                                      vfInterpolator,pointMask2)

        simMeasurer1 = similarityMeasure.ssd(points1, evalPointsIntensitySegment1, fixedImage,
                                             intensityInterpolator, coef=simCoef ,pointsMask=pointMask1, pointsMaskLabel=1)
        simMeasurer2 = similarityMeasure.ssd(points2, evalPointsIntensitySegment2, fixedImage,
                                             intensityInterpolator, coef=simCoef,pointsMask=pointMask2, pointsMaskLabel=1)

        reg = []


        ###FFD
        """c1 = [simMeasurer1]
        c2 = [simMeasurer2]"""

        ###CCDIR
        origDis1 = points1[pointMask1 == 0][selfColSel1 == 1].clone() * 0
        origDis2 = points2[pointMask2 == 0][selfColSel2 == 1].clone() * 0
        #(pts,sdf,interpolator,simplices,verticesUndeformed,coef,device,pointsMask,pointsMaskLabel,unmaskedDis,vertMask,vertMaskLabel)

        sc1 = collisionDetection.selfintersectionDetection(points1, sdf, intensityInterpolator,  simp,
                                                           verticesOriginal, colCoef,
                                                           device, pointsMask=pointMask1, pointsMaskLabel=0,
                                                           unmaskedDis=origDis2, vertMask=selfColSel1,vertMaskLabel=1)
        sc2 = collisionDetection.selfintersectionDetection(points2, sdf, intensityInterpolator,  simp,
                                                           verticesOriginal, colCoef,
                                                           device, pointsMask=pointMask2, pointsMaskLabel=0,
                                                           unmaskedDis=origDis1,
                                                           vertMask=selfColSel2,vertMaskLabel=1)
        sc2.siPartner = sc1
        sc1.siPartner = sc2

        sc1.name = segmentTuple[0]
        sc2.name = segmentTuple[1]

        c1 = [simMeasurer1, sc1]
        c2 = [simMeasurer2, sc2]


        optimizer1 = optimization.gradientDescentBacktracking([decisionVariablesSegment1], lr=0.001, max_iters=5)
        optimizer2 = optimization.gradientDescentBacktracking([decisionVariablesSegment2], lr=0.001, max_iters=5)

        rho = 10
        closure1 = optimization.closureADMM(optimizer1, nrDeformationSegment1, mainTerm=c1, regTerms=reg, rho=rho)
        closure2 = optimization.closureADMM(optimizer2, nrDeformationSegment2, mainTerm=c2, regTerms=reg, rho=rho)



        consti = optimization.constrainerLagrangianADMM(torch.zeros(shared1.size(), device=device),
                                               torch.zeros(shared1.size(), device=device), 1, -1, nrDeformationSegment1, shared1)
        dict = optimization.algorithmADMM( 10, 2, consti, optimizer1, optimizer2, closure1,
                                                          closure2, decisionVariablesSegment1,
                                                          decisionVariablesSegment2, rho)

        evalPointsSegment1 = nrDeformationSegment1.apply(evalPointsSegment1, decisionVariablesSegment1.data.detach())
        evalPointsSegment2 = nrDeformationSegment2.apply(evalPointsSegment2, decisionVariablesSegment2.data.detach())
        shared1 = nrDeformationSegment1.apply(shared1, decisionVariablesSegment1.data.detach())
        shared2 = nrDeformationSegment2.apply(shared2, decisionVariablesSegment2.data.detach())
        v1 = nrDeformationSegment1.apply(vertices[selfColSel1 == 1], decisionVariablesSegment1.data.detach())
        v2 = nrDeformationSegment2.apply(vertices[selfColSel2 == 1], decisionVariablesSegment2.data.detach())
        vertices[selfColSel1 == 1] = v1
        vertices[selfColSel2 == 1] = v2
        print("differences shared sum",torch.sum(torch.norm(shared2-shared1,dim=1)))
        print("differences shared mean",torch.mean(torch.norm(shared2-shared1,dim=1)))
        print("differences shared std",torch.std(torch.norm(shared2-shared1,dim=1)))


        # PIECEWISE DEFORMATION: Save x where needed fx
        np.save(os.path.join(workingDirectory, "results",
                             f"transformation_nonrigid_segment_{segmentTuple[0]}_scale_" + str(scale.cpu())),
                decisionVariablesSegment1.data.detach().cpu())

        np.save(os.path.join(workingDirectory, "results",
                             f"transformation_nonrigid_segment_{segmentTuple[1]}_scale_" + str(scale.cpu())),
                decisionVariablesSegment2.data.detach().cpu())

        with open(os.path.join(workingDirectory, "results",
                               f"transformation_nonrigid_segment_{segmentTuple}_scale_"+ str(scale.cpu()) + ".json"), "w") as outfile:
            json.dump(dict, outfile)
        with open(os.path.join(workingDirectory, "results",
                               f"class_transformation_nonrigid_segment_{segmentTuple[0]}_scale_"+str(scale.cpu()) + ".pkl"),
                  'wb') as output_file:
            pickle.dump(nrDeformationSegment1, output_file)
        with open(os.path.join(workingDirectory, "results",
                               f"class_transformation_nonrigid_segment_{segmentTuple[1]}_scale_"+str(scale.cpu())+ ".pkl"),
                  'wb') as output_file:
            pickle.dump(nrDeformationSegment2, output_file)
    return shared1,shared2

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    workingDirectory = os.path.join(current_dir, "syntheticData/selfintersection2D")
    device = "cuda:0"
    voxelToMmRatio = torch.tensor([1.,1.], device=device)

    if os.path.exists(os.path.join(workingDirectory, 'results/')):
        pass
    else:
        os.mkdir(os.path.join(workingDirectory, 'results/'))
    files = glob.glob(os.path.join(workingDirectory, 'results/*'))
    for f in files:
        path, name = os.path.split(f)
        if name.startswith("class_transformation") or name.startswith("transformation"):
            os.remove(f)

    segmentTuples = [(1, 2)]
    for segmentTuple in segmentTuples:



        shared1, shared2=deformable_ccdir(device, workingDirectory, segmentTuple, simCoef=1, lr=0.0001,colCoef=10000)

        #postprocessing.visual_convergence(workingDirectory,segmentTuple)
        #postprocessing.visual_grid(device, workingDirectory, voxelToMmRatio, segmentTuple)
        fieldsDet = postprocessing.measure_jacobian(device, workingDirectory, voxelToMm=voxelToMmRatio,segmentsOfInterest=segmentTuple)
        print("detJac")
        for key in fieldsDet:
            f = fieldsDet[key]
            f = np.array(f)
            print(key, np.sum(np.where(f <= 0, 1, 0)))
        dice=postprocessing.measure_dice(device, workingDirectory, voxelToMm=voxelToMmRatio,segmentsOfInterest=segmentTuple)
        print("dice",dice)
        icMean, icStd, icTensor = postprocessing.measure_inverseConsistency(device, workingDirectory,voxelToMm=voxelToMmRatio,segmentsOfInterest=segmentTuple)
        print("ic", icMean, icStd)

        shearBoundariesDict = postprocessing.measure_shear(device, workingDirectory, voxelToMm=voxelToMmRatio,segmentsOfInterest=segmentTuple)
        postprocessing.measure_selfIntersection2dContourBased(device,workingDirectory,voxelToMm=voxelToMmRatio,segmentsOfInterest=segmentTuple)

        #postprocessing.visual_pullback(device, workingDirectory, voxelToMmRatio, segmentTuple)
        #postprocessing.visual_shear(device, workingDirectory, voxelToMmRatio, segmentTuple)
        postprocessing.visual_vector(device, workingDirectory, voxelToMmRatio, segmentTuple)


main()






