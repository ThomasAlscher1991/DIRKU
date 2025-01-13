import torch
import numpy as np
import os, glob, tqdm
import json
import pickle
from dirku import utils, interpolation, geometricTransformations, similarityMeasure, optimization, \
    numericalIntegration, collisionDetection, regularization,postprocessing

def deformable_piecewise(device, workingDirectory, segment,voxelToMmRatio,simCoef=1,colCoef=1,dilation=None):
    # BASICS: load images
    movingImage = torch.unsqueeze(torch.from_numpy(np.load(os.path.join(workingDirectory, "moving.npy"))),
                                  dim=0).to(device=device)
    fixedImage = torch.unsqueeze(torch.from_numpy(np.load(os.path.join(workingDirectory, "fixed.npy"))),
                                 dim=0).to(device=device)
    fixedImageMask = torch.unsqueeze(
        torch.from_numpy(np.load(os.path.join(workingDirectory, "fixed_mask.npy"))), dim=0).to(device=device)
    movingImageMask = torch.unsqueeze(
        torch.from_numpy(np.load(os.path.join(workingDirectory, "moving_mask.npy"))), dim=0).to(device=device)

    max = torch.max(torch.stack([movingImage, fixedImage]))

    movingImage = (movingImage / max) * 1
    fixedImage = (fixedImage / max) * 1

    s = utils.sdfCreator(device,reuse=False,workingDirectory=workingDirectory,segmentName=segment)
    mask1=torch.where(fixedImageMask==segment,0,1)
    mask0=torch.where(fixedImageMask==0,0,1)
    mask=mask1*mask0
    sdf=s.fromMask(torch.where(mask==1,1,0),voxelSizes=voxelToMmRatio)

    intensityInterpolator = interpolation.cubic(device, torch.tensor([1., 1.], device=device))

    evalPoints = utils.getEvaluationPoints(device,movingImage,mask=movingImageMask,maskLabel=segment,dilation=None)
    evalPointsIntensity,_,_=intensityInterpolator(evalPoints,movingImage)


    scales = torch.tensor([[10., 10.], [5., 5.], [1., 1.]], device=device)

    torch.cuda.empty_cache()
    for scale in scales:
        tqdm.tqdm.write(f"Scale {scale}")
        decisionVariablesReg = utils.getGridPoints(movingImageMask, scale, 10)
        decisionVariablesReg.requires_grad = True

        vfInterpolator = interpolation.cubic(device, scale)

        integrator = numericalIntegration.trapezoidal(10, corrector_steps=10, stationary=False)
        nrDeformationReg = geometricTransformations.nonrigidDeformation(evalPoints, integrator,
                                                                        vfInterpolator)

        simMeasurer = similarityMeasure.ssd(evalPoints, evalPointsIntensity, fixedImage,
                                            intensityInterpolator, coef=simCoef)

        ###LDDMM

        """c1=[simMeasurer]
        reg1 =[]
        optimizer1 = optimization.gradientDescentBacktracking([decisionVariablesReg], lr=0.001, max_iters=5)
        closure1 = optimization.closureGradientDescent(optimizer1, nrDeformationReg, mainTerms=c1, regTerms=reg1)
        dict=optimization.algorithmGradientDescent(10,optimizer1,closure1,decisionVariablesReg)"""


        ##CCDIR
        decisionVariablesCol = utils.getGridPoints(movingImageMask, scale, 10)
        decisionVariablesCol.requires_grad = True

        nrDeformationCol = geometricTransformations.nonrigidDeformation(evalPoints, integrator,
                                                                     vfInterpolator)

        colDet = collisionDetection.intersectionDetection(evalPoints,sdf,intensityInterpolator,colCoef)

        c1=[simMeasurer]
        c2=[colDet]
        reg1 =[]
        reg2 =[]

        optimizer1 = optimization.gradientDescentBacktracking([decisionVariablesReg], lr=0.001, max_iters=5)
        optimizer2 = optimization.gradientDescentBacktracking([decisionVariablesCol], lr=0.001, max_iters=5)

        rho = 0.5

        closure1 = optimization.closureADMM(optimizer1, nrDeformationReg, mainTerms=c1, regTerms=reg1, rho=rho)
        closure2 = optimization.closureADMM(optimizer2, nrDeformationCol, mainTerms=c2, regTerms=reg2, rho=rho)

        consti = optimization.constrainerEulerianADMM(torch.zeros(decisionVariablesCol.size(),device=device),torch.zeros(decisionVariablesCol.size(),device=device),1,-1)
        dict = optimization.algorithmADMM( 10, 2, consti, optimizer1, optimizer2, closure1,
                                                          closure2, decisionVariablesReg,
                                                          decisionVariablesCol, rho)


        evalPoints = nrDeformationReg.apply(evalPoints, decisionVariablesReg.data.detach())

        np.save(os.path.join(workingDirectory, "results",
                             f"transformation_nonrigid_segment_{segment}_scale_" + str(scale.cpu())),
                decisionVariablesReg.data.detach().cpu())
        with open(os.path.join(workingDirectory, "results",
                               f"transformation_nonrigid_segment_{segment}_scale_"+ str(scale.cpu()) + ".json"), "w") as outfile:
            json.dump(dict, outfile)
        with open(os.path.join(workingDirectory, "results",
                               f"class_transformation_nonrigid_segment_{segment}_scale_"+str(scale.cpu()) + ".pkl"),
                  'wb') as output_file:
            pickle.dump(nrDeformationReg, output_file)
        np.save(os.path.join(workingDirectory, "results",
                             f"transformation_nonrigid_segment_{segment}_scale_" + str(scale.cpu())),
                decisionVariablesReg.data.detach().cpu())
        with open(os.path.join(workingDirectory, "results",
                               f"transformation_nonrigid_segment_{segment}_scale_" + str(scale.cpu()) + ".json"),
                  "w") as outfile:
            json.dump(dict, outfile)
        with open(os.path.join(workingDirectory, "results",
                               f"class_transformation_nonrigid_segment_{segment}_scale_" + str(scale.cpu()) + ".pkl"),
                  'wb') as output_file:
            pickle.dump(nrDeformationReg, output_file)

def test_main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    workingDirectory = os.path.join(current_dir, "syntheticData/intersection2D")
    device = "cpu"
    voxelToMm = torch.tensor([1., 1.], device=device)


    if os.path.exists(os.path.join(workingDirectory, 'results/')):
        pass
    else:
        os.mkdir(os.path.join(workingDirectory, 'results/'))
    files = glob.glob(os.path.join(workingDirectory, 'results/*'))
    for f in files:
        path, name = os.path.split(f)
        if name.startswith("class_transformation") or name.startswith("transformation"):
            os.remove(f)

    for segment in [1,2]:
        pass
        deformable_piecewise(device, workingDirectory, segment, voxelToMm, simCoef=100000, colCoef=21500000,dilation=None)
    dice=postprocessing.measure_dice(device,workingDirectory,segmentsOfInterest=[1,2])
    print("dice",dice)
    fields=postprocessing.measure_jacobian(device,workingDirectory,voxelToMm,segmentsOfInterest=[1,2])
    print("detJac")
    for key in fields:
        f=fields[key]
        f=np.array(f)
        print(key,np.sum(np.where(f<=0,1,0)))
    icMean,icStd,_=postprocessing.measure_inverseConsistency(device,workingDirectory,voxelToMm,segmentsOfInterest=[1,2])
    print("ic",icMean,icStd)
    dictIntersection=postprocessing.measure_intersection2d(device,workingDirectory,segmentsOfInterest=[1,2])
    print("intersection2D")
    for i in dictIntersection:
        print(i, dictIntersection[i])
    #postprocessing.measure_shear(device,workingDirectory,segmentsOfInterest=[1,2])
    #postprocessing.visual_convergence(workingDirectory,[1,2])
    #postprocessing.visual_grid(device,workingDirectory,voxelToMm,[1,2])
    #postprocessing.visual_pullback(device,workingDirectory,voxelToMm,[1,2])
    #postprocessing.visual_shear(device,workingDirectory,voxelToMm,[1,2])
    #postprocessing.visual_vector(device,workingDirectory,voxelToMm,[1,2])








