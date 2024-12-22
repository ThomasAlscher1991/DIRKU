import os.path
import igl
import torch
import numpy as np
import skfmm
from typing import Optional, Type, Union, Tuple
from torch import Tensor

class sdfCreator:
    """Class for creating signed distance fields (SDF).
    Voxels or pixels equating to maskLabel in mask are considered inside an object.
    :param device: computation device, see torch docs
    :type device: str
        :param reuse: whether the SDF should be reused
        :type reuse: bool
        :param workingDirectory: directory of the registration data
        :type workingDirectory: str
        :param segmentName: the label of the object to SDF is for
        :type segmentName: int
        """

    def __init__(self, device: str, reuse: Optional[bool]=False, workingDirectory: Optional[bool]=None, segmentName: Optional[int]=None):
        "Constructor method."
        self.segmentName = segmentName
        self.reuse = reuse
        self.workingDirectory = workingDirectory
        self.device = device

    def checkReuse(self)->bool:
        """Checks if SDF should be reused.
        If so, checks the reuse folder.
        :return: answer whether SDF was found or not
        :rtype: bool"""
        self.checkReuseFolder()
        print("reuse sdf")
        if os.path.exists(os.path.join(self.workingDirectory, "reuse", f"sdf_segment_{self.segmentName}.npy")):
            print("found")
            return True
        else:
            print(" not found")
            return False

    def loadReuse(self)->Tensor:
        """Reuses old SDF.
        :return: SDF
        :rtype: torch.Tensor"""
        return torch.from_numpy(
            np.load(os.path.join(self.workingDirectory, "reuse", f"sdf_segment_{self.segmentName}.npy"))).to(
            device=self.device)

    def saveReuse(self, sdf):
        """Saves SDF for reuse."""
        np.save(os.path.join(self.workingDirectory, "reuse", f"sdf_segment_{self.segmentName}.npy"),
                sdf.cpu().numpy())

    def fromMask(self, mask: Tensor, voxelSizes: Optional[Tensor]=None, invert: Optional[bool]=False)->Tensor:
        """Creates a SDF from a masked image. Entries with 1 will be assigned outside, everything else inside.
        :param mask: mask of the moving image (1,dim1,dim2(,dim3))
        :type mask: torch.Tensor
        :param voxelSizes: cell dimensions in mm
        :type voxelSizes: torch.Tensor
        :param invert: flip ones and non-ones integers in mask
        :type invert: bool"""
        if voxelSizes is None:
            if len(mask.size())==3:
                voxelSizes=torch.tensor([1.,1.])
            else:
                voxelSizes=torch.tensor([1.,1.,1.])
        else:
            pass
        if self.reuse:
            if self.checkReuse():
                return self.loadReuse()
            else:
                if invert:
                    mask = torch.where(mask == 1, -1., 1.)
                else:
                    mask = torch.where(mask == 1, 1., -1.)
                sdf = skfmm.distance(mask.cpu()[0], dx=voxelSizes.cpu())
                sdf = np.where(sdf > 0., 0, sdf)
                sdf = torch.unsqueeze(torch.from_numpy(sdf).to(device=self.device),dim=0)
                self.saveReuse(sdf)
                return sdf
        else:
            if invert:
                mask = torch.where(mask == 1, -1., 1.)
            else:
                mask = torch.where(mask == 1, 1., -1.)
            sdf = skfmm.distance(mask.cpu()[0], dx=voxelSizes.cpu())
            sdf = np.where(sdf > 0., 0, sdf)
            sdf = torch.unsqueeze(torch.from_numpy(sdf).to(device=self.device), dim=0)
            return sdf

    def fromMesh(self, pathToMesh: str, domain: Tensor)->Tensor:
        """Create a SDF from a 3D triangle surface mesh (stl files).
        :param pathToMesh: path to file
        :type pathToMesh: str
        :param domain: image domain (1,dim1,dim2,dim3)
        :type domain: torch.Tensor
        l"""
        vertices, faces = igl.read_triangle_mesh(pathToMesh, 'float')
        if self.reuse:
            if self.checkReuse():
                return self.loadReuse()
            else:
                indices = np.indices(domain.cpu()[0].size())
                pts = np.empty((np.prod(domain.cpu().size()), len(domain[0].cpu().size())))
                for i, slide in enumerate(indices):
                    pts[:, i] = slide.flatten()
                s, i, c = igl.signed_distance(pts, vertices, faces)
                ptsInt = pts.astype(int)
                sdf = domain.cpu().numpy().copy() * 0
                sdf[0,ptsInt[:, 0], ptsInt[:, 1], ptsInt[:, 2]] = s
                sdf = torch.from_numpy(sdf).to(device=self.device)
                self.saveReuse(sdf)
                return sdf
        else:
            indices = np.indices(domain.cpu()[0].size())
            pts = np.empty((np.prod(domain.cpu().size()), len(domain[0].cpu().size())))
            for i, slide in enumerate(indices):
                pts[:, i] = slide.flatten()
            s, i, c = igl.signed_distance(pts, vertices, faces)
            ptsInt = pts.astype(int)
            sdf = domain.cpu().numpy().copy() * 0
            sdf[0, ptsInt[:, 0], ptsInt[:, 1], ptsInt[:, 2]] = s
            sdf = torch.from_numpy(sdf).to(device=self.device)
            return sdf

    def checkReuseFolder(self):
        """Checks if a reuse folder is already created. If not, creates it."""
        if os.path.exists(os.path.join(self.workingDirectory, 'reuse/')):
            pass
        else:
            os.mkdir(os.path.join(self.workingDirectory, 'reuse/'))


    def getGrads(self, sdf: Tensor)->Tuple[Tensor,Tensor,Optional[Tensor]]:
        """Returns the gradients of SDF.
            :param sdf: signed distance field
            :type sdf: torch.Tensor
            :return gradX,gradY,gradZ: gradients
            :rtype sdf: list of torch.Tensor"""
        if len(sdf.size()) == 4:

            translated = sdf.cpu().numpy()[0]
            gradX, gradY, gradZ = np.gradient(translated)
            gradX = torch.unsqueeze(torch.from_numpy(gradX).to(device=sdf.device), dim=0)
            gradY = torch.unsqueeze(torch.from_numpy(gradY).to(device=sdf.device), dim=0)
            gradZ = torch.unsqueeze(torch.from_numpy(gradZ).to(device=sdf.device), dim=0)
            return [gradX, gradY, gradZ]

        elif len(sdf.size()) == 3:
            translated = sdf.cpu().numpy()[0]
            gradX, gradY = np.gradient(translated)
            gradX = torch.unsqueeze(torch.from_numpy(gradX).to(device=sdf.device), dim=0)
            gradY = torch.unsqueeze(torch.from_numpy(gradY).to(device=sdf.device), dim=0)
            return [gradX, gradY]
        else:
            raise Exception("check dimensions sdf")
