import os.path
import igl
import torch
import numpy as np
import skfmm


class sdfCreator:
    """Class for creating signed distance fields.
    Voxels equating to maskLabel in mask are considered inside an object.
    Invert for self-intersections.
        :param mask: mask of the moving image (dim1,dim2(,dim3))
        :type mask: torch.tensor
        :param maskLabel: label of segments considered inside
        :type maskLabel: torch.tensor
        :param voxelSizes: cell dimensions in mm
        :type voxelSizes: list of floats
        :param voxelSizes: invert true will assume all voxels equating to maskLabel in mask are considered outside an object.
        :type voxelSizes: boolean
        """

    def __init__(self, device, reuse=False, workingDirectory=None, segmentName=None):
        "Constructor method"
        self.segmentName = segmentName
        self.reuse = reuse
        self.workingDirectory = workingDirectory
        self.device = device

    def checkReuse(self):
        self.checkReuseFolder()
        print("reuse sdf")
        if os.path.exists(os.path.join(self.workingDirectory, "reuse", f"sdf_segment_{self.segmentName}.npy")):
            print("found")
            return True
            found = True
        else:
            print(" not found")
            return False

    def loadReuse(self):
        return torch.from_numpy(
            np.load(os.path.join(self.workingDirectory, "reuse", f"sdf_segment_{self.segmentName}.npy"))).to(
            device=self.device)

    def saveReuse(self, sdf):
        np.save(os.path.join(self.workingDirectory, "reuse", f"sdf_segment_{self.segmentName}.npy"),
                sdf.cpu().numpy())

    def fromMask(self, mask, voxelSizes=None, invert=False):
        """from stl mesh in 3d"""
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

    def fromMesh(self, pathToMesh, domain):
        """from stl mesh in 3d"""
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
        if os.path.exists(os.path.join(self.workingDirectory, 'reuse/')):
            pass
        else:
            os.mkdir(os.path.join(self.workingDirectory, 'reuse/'))


    def getGrads(self, sdf):
        """Returns thegradients of signed distance field.
            :param sdf: signed distance field
            :type sdf: torch.tensor
            :return gradX,gradY,gradZ: gradients
            :rtype sdf: lsit of torch.tensor"""
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
