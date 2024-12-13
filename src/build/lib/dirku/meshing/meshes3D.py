import pyvista as pv
import torch
from skimage import measure
import numpy as np
from stl import mesh #pip install numpy-stl
import igl
import wildmeshing as wm
import os
import pymeshlab
import sys
import matplotlib.pyplot as plt



class surfaceMesh():
    """Creates a triangle surface mesh in 3D.
    Meshes can be saved for reuse.
    Segments assigned segmentLabel in mask are considered inside the object.
        :param mask: mask of the moving image (1,dim1,dim2,dim3)
        :type mask: torch.tensor
        :param segmentLabel: label of segmentation to be meshed
        :type segmentLabel: int
        :param device: torch device
        :type device: string
        :param workingDirectory: path to working directory, see docs
        :type workingDirectory: string
        :param faceNumbers: number of faces required; if None no reduction in faces
        :type faceNumbers: int
        :param reuse: if false, mesh will be generated anew on every call; if true, older meshes will be loaded
        :type reuse: boolean
        :param level: level of iso surface for marching cubes
        :type level: int"""
    def __init__(self, mask, segmentName, device, workingDirectory=None, faceNumbers=None, reuse=False, level=0.8):
        "Constructor method"
        self.workingDirectory = workingDirectory
        self.faceNumbers = faceNumbers
        self.device = device
        self.segmentName = segmentName
        self.mask = mask.cpu().numpy()
        self.reuse = reuse
        self.level = level


    def checkReuseFolder(self):
        """Checks the workingDirectory for the reuse folder"""
        if os.path.exists(os.path.join(self.workingDirectory, 'reuse/')):
            pass
        else:
            os.mkdir(os.path.join(self.workingDirectory, 'reuse/'))

    def triangulate(self):
        """Creates a triangle surface mesh."""
        mask = np.where(self.mask == 1, 1, 0)
        vertices, faces, normals, values = measure.marching_cubes(mask, level=self.level)  # level was 0
        faces_new = np.array(faces.copy())
        faces_new[:, 1] = faces[:, 2]
        faces_new[:, 2] = faces[:, 1]
        faces = faces_new
        cube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
        for i, f in enumerate(faces):
            for j in range(3):
                cube.vectors[i][j] = vertices[f[j], :]
        cube.save(os.path.join(self.workingDirectory, "reuse", f"surface_mesh_segment_{self.segmentName}.stl"))
        if self.faceNumbers is not None:
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(
                os.path.join(self.workingDirectory, "reuse", f"surface_mesh_segment_{self.segmentName}.stl"))
            ms.meshing_decimation_quadric_edge_collapse(targetfacenum=self.faceNumbers)
            ms.save_current_mesh(
                os.path.join(self.workingDirectory, "reuse", f"surface_mesh_segment_{self.segmentName}.stl"))
        else:
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(
                os.path.join(self.workingDirectory, "reuse", f"surface_mesh_segment_{self.segmentName}.stl"))
            ms.meshing_decimation_quadric_edge_collapse()
            ms.save_current_mesh(
                os.path.join(self.workingDirectory, "reuse", f"surface_mesh_segment_{self.segmentName}.stl"))
        vertices, faces = igl.read_triangle_mesh(
            os.path.join(self.workingDirectory, "reuse", f"surface_mesh_segment_{self.segmentName}.stl"), 'float')
        return vertices, faces

    def getVerticesAndSimplicesSurface(self):
        """Returns the vertices and simplices of the surface mesh.
            :return vertices: vertices (# vertices, 3)
            :rtype vertices: torch.tensor
            :return simplices: simplices (#simplices,3)
            :rtype simplices: torch.tensor"""
        if self.reuse:
            self.checkReuseFolder()
            print("reuse surface")
            if os.path.exists(
                    os.path.join(self.workingDirectory, "reuse", f"surface_mesh_segment_{self.segmentName}.stl")):
                print("found")
                vertices, faces = igl.read_triangle_mesh(
                    os.path.join(self.workingDirectory, "reuse", f"surface_mesh_segment_{self.segmentName}.stl"),
                    'float')
            else:
                print("not found")
                vertices, faces = self.triangulate()
        else:
            vertices, faces = self.triangulate()
        return torch.from_numpy(vertices).to(device=self.device).float(), torch.from_numpy(faces).to(
            device=self.device).float()


class tetrahedralMesh():
    """Creates a tetrahedral mesh in 3D.
    Can create the mesh either from a segmented mask image, or an existing 3D stl surface mesh.
    If creating from a segmented mask image, will create the surface mesh on the fly.
    Meshes can be saved for reuse.
    Segments assigned segmentLabel in mask are considered inside the object.
        :param device: torch device
        :type device: string
        :param workingDirectory: path to working directory, see docs
        :type workingDirectory: string
        :param reuse: if false, mesh will be generated anew on every call; if true, older meshes will be loaded
        :type reuse: boolean
    """
    def __init__(self,device,workingDirectory=None,reuse=False):
        "Constructor method"
        self.workingDirectory=workingDirectory
        self.device=device
        self.reuse=reuse

    def getTetrahedralMeshFromMask(self,mask,segmentName,faceNumbers=None,level=0.8):
        """Creates a tetrahedral mesh from a mask image. Creates a surface mesh first.
        :param mask: mask of the moving image (1,dim1,dim2,dim3)
        :type mask: torch.tensor
        :param segmentLabel: label of segmentation to be meshed
        :type segmentLabel: int
        :param faceNumbers: number of faces required; if None no reduction in faces
        :type faceNumbers: int
        :param level: level of iso surface for marching cubes
        :type level: int
        :return volVertices: vertices (#simplices,3)
        :rtype volVertices: torch.tensor
        :return volTets: simplices (#simplices,4)
        :rtype volTets: torch.tensor"""
        c=surfaceMesh(mask, segmentName, self.device, self.workingDirectory, faceNumbers, self.reuse, level)
        vertices,faces=c.getVerticesAndSimplicesSurface()
        vertices=vertices.cpu().numpy()
        faces=faces.cpu().numpy()
        if self.reuse:
            print("reuse tetrahedral mesh")
            self.checkReuseFolder()
            path = os.path.join(self.workingDirectory, "reuse", f"tetrahedral_mesh_segment_{segmentName}.vtk")
            if os.path.exists(path):
                print("found")
                loaded_mesh = pv.read(path)
                volVertices = loaded_mesh.points
                volSimplices = loaded_mesh.cells.reshape(-1, 5)[:, 1:]
            else:
                print("not found")
                volVertices, volSimplices = self.tetrahedralize(vertices, faces)
                cells = np.hstack([np.full((volSimplices.shape[0], 1), 4), volSimplices])
                cell_types = np.full(volSimplices.shape[0], pv.CellType.TETRA)
                grid = pv.UnstructuredGrid(cells, cell_types, volVertices)
                grid.save(path, binary=True)
        else:
            volVertices, volSimplices = self.tetrahedralize(vertices, faces)
        return torch.from_numpy(volVertices).to(device=self.device).float(),torch.from_numpy(volSimplices).to(device=self.device).float()

    def getTetrahedralMeshFromSurface(self,pathToMesh,segmentName):
        """Creates a tetrahedral mesh from a triangel surface.
        :param pathToMesh: path to stl mesh
        :type pathToMesh: str
        :param segmentLabel: label of segmentation to be meshed
        :type segmentLabel: int
        :return volVertices: vertices (#simplices,3)
        :rtype volVertices: torch.tensor
        :return volTets: simplices (#simplices,4)
        :rtype volTets: torch.tensor"""
        vertices, faces = igl.read_triangle_mesh(
            os.path.join(pathToMesh), 'float')
        if self.reuse:
            print("reuse tetrahedral mesh")
            self.checkReuseFolder()
            path=os.path.join(self.workingDirectory,"reuse",f"tetrahedral_mesh_segment_{segmentName}.vtk")
            if os.path.exists(path):
                print("found")
                loaded_mesh = pv.read(path)
                volVertices = loaded_mesh.points
                volSimplices = loaded_mesh.cells.reshape(-1, 5)[:, 1:]
            else:
                print("not found")
                volVertices, volSimplices = self.tetrahedralize(vertices, faces)
                cells = np.hstack([np.full((volSimplices.shape[0], 1), 4), volSimplices])
                cell_types = np.full(volSimplices.shape[0], pv.CellType.TETRA)
                grid = pv.UnstructuredGrid(cells, cell_types, volVertices)
                grid.save(path, binary=True)
        else:
            volVertices, volSimplices = self.tetrahedralize(vertices, faces)
        return torch.from_numpy(volVertices.copy()).to(device=self.device).float(),torch.from_numpy(volSimplices.copy()).to(device=self.device).float()

    def checkReuseFolder(self):
        """Checks the workingDirectory for the reuse folder"""
        if os.path.exists(os.path.join(self.workingDirectory, 'reuse/')):
            pass
        else:
            os.mkdir(os.path.join(self.workingDirectory, 'reuse/'))

    def tetrahedralize(self,vertices,simplices):
        """Creates a tetrahedral mesh.
            :return volVertices: vertices (#simplices,3)
            :rtype volVertices: torch.tensor
            :return volTets: simplices (#simplices,4)
            :rtype volTets: torch.tensor"""
        tetra = wm.Tetrahedralizer()
        tetra.set_mesh(vertices, simplices)
        tetra.tetrahedralize()
        volVertices, volTets = tetra.get_tet_mesh()
        volVertices, volTets, _, _ = igl.remove_unreferenced(volVertices, volTets)
        return volVertices, volTets

    def getExteriorPoints(self,vertices,simplices):
         """Returns the exterior vertices of the tetrahedra mesh.
            :param vertices: vertices (# vertices, 3)
            :type vertices: torch.tensor
            :param workingDirectory: simplices (#simplices,4)
            :type workingDirectory: torch.tensor
            :return vertices: vertices (# vertices, 3)
            :rtype vertices: torch.tensor"""
         vertices=vertices.cpu().numpy
         simplices=simplices.cpu().numpy
         cells = np.hstack([np.full((simplices.shape[0], 1), 4), simplices]).flatten()
         cell_types = np.full(simplices.shape[0], pv.CellType.TETRA, dtype=np.uint8)
         grid = pv.UnstructuredGrid(cells, cell_types, vertices)
         surface = grid.extract_surface()
         exterior_points = surface.points
         return torch.from_numpy(exterior_points).to(devive=self.device)


