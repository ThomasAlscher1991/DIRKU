import torch
class selfintersectionDetection:
    """ Class for self-collision detection based on meshes.
        This self-collision detection implementation calculates the self-collisons at t=1 of deformation.
        Should be used as main term  in ADMM.
        :param pts: points used for collision detection
        :type pts: torch.tensor
        :param sdf: signed distance map of scene with object that should avoid self-intersection2D
        :type sdf: torch.tensor
        :param interpolator: CUDA device or cpu, see torch docs
        :type interpolator: interpolation class
        :param grads: gradients of sdf [x,y,z]
        :type grads: list of torch.tensors
        :param simplices: simplices of mesh (either triangles or tetrahedra); (#of simplices,3 or 4)
        :type simplices: torch.tensor
        :param vertices: vertices of mesh (# of vertices, 2 or 3)
        :type vertices: torch.tensor
        :param coef: coefficient applied to the collision loss
        :type coef: float
        """
    def __new__(cls, pts,sdf,interpolator,grads,simplices,verticesUndeformed,coef,device,pointsMask=None,pointsMaskLabel=None,unmaskedDis=None,vertMask=None):
        """ CConstructor method. Decides dimensionality.
        """
        if pts.size(1) == 2:
            return selfintersectionDetection2D(pts,sdf,interpolator,grads,simplices,verticesUndeformed,coef,device,pointsMask,pointsMaskLabel,unmaskedDis,vertMask)
        elif pts.size(1) == 3:
            return selfintersectionDetection3D(pts,sdf,interpolator,grads,simplices,verticesUndeformed,coef,device,pointsMask,pointsMaskLabel,unmaskedDis,vertMask)
        else:
            raise ValueError("Unsupported dimension. Only 2D and 3D are supported.")


class selfintersectionDetection2D():
    """ Class for self-collision detection based on meshes.
    Implementation for point-tetrahedron check is taken from https://stackoverflow.com/questions/25179693/how-to-check-whether-the-point-is-in-the-tetrahedron-or-not.
    :param pts: points used for collision detection
    :type pts: torch.tensor
    :param sdf: signed distance map of scene with object that should avoid self-intersection2D
    :type sdf: torch.tensor
    :param interpolator: CUDA device or cpu, see torch docs
    :type interpolator: interpolation class
    :param grads: gradients of sdf [x,y,z]
    :type grads: list of torch.tensors
    :param simplices: simplices of mesh (either triangles or tetrahedra); (#of simplices,3 or 4)
    :type simplices: torch.tensor
    :param vertices: vertices of mesh (# of vertices, 2 or 3)
    :type vertices: torch.tensor
    :param coef: coefficient applied to the collision loss
    :type coef: float
    :param device: sets the computation device, see torch
    :type device: string
    """
    def __init__(self,pts,sdf,interpolator,grads,simplices,verticesUndeformed,coef,device,pointsMask,pointsMaskLabel,unmaskedDis,vertMask):
        """constructor method"""
        self.sdf=sdf
        self.intP=interpolator
        self.grads=grads #has to have the following form (dimension,x,y,z)
        self.simplices=simplices
        self.verticesUndeformed=verticesUndeformed
        self.device=device
        self.pts=pts
        self.coef=coef
        self.pointsMask=pointsMask
        self.pointsMaskLabel=pointsMaskLabel
        self.unmaskedDis=unmaskedDis
        self.vertMask=vertMask
        self.siPartner=None



    def pointsMaskSelection(self,dis):
        nodesMoved =self.pts[self.pointsMask == self.pointsMaskLabel].clone()
        dis = dis[self.pointsMask == self.pointsMaskLabel]
        return dis,nodesMoved
    def vertMaskSelection(self,dis,nodesMoved):
        nodesMoved[self.vertMask == 1] = nodesMoved[self.vertMask == 1] + dis[self.vertMask == 1]
        nodesMoved[self.vertMask == 0] = nodesMoved[self.vertMask == 0] + self.unmaskedDis
        self.siPartner.unmaskedDis = dis[self.vertMask == 1].data
        return nodesMoved

    def __call__(self,dis=0,**kwargs):
        """Executing 2D self collision detection.
        :param dis: displacement of pts
        :type dis: torch.tensor
        :return: summed depth of selfintersecting points
        :rtype: torch.tensor"""
        if self.pointsMask is not None:
            nodesMoved=self.pts[self.pointsMask==self.pointsMaskLabel].clone()
            #nodesMoved=self.nodesOrig.clone()
            dis=dis[self.pointsMask==self.pointsMaskLabel]
            nodesMoved[self.vertMask==1]=nodesMoved[self.vertMask==1]+dis[self.vertMask==1]
            nodesMoved[self.vertMask==0]=nodesMoved[self.vertMask==0]+self.unmaskedDis
            self.siPartner.unmaskedDis=dis[self.vertMask==1].data

        else:
            nodesMoved=self.pts+dis
        l=self.loss2D(nodesMoved)*self.coef
        return l



    def barycentricToCartesian2D(self,barycentricCoords, e1,e2,e3):
        """Transformation from barycentric to cartesian coordinates in 2D.
            :param barycentricCoords: barycentric coordinates
            :type barycentricCoords: torch.tensor
            :param e1,e2,e3: cartesian coordinates of simplex
            :type e1,e2,e3: torch.tensor
            :return: cartesian coordinates
            :rtype: torch.tensor"""
        p=torch.unsqueeze(e1,dim=1)*barycentricCoords[:,0:2]+torch.unsqueeze(e2,dim=1)*barycentricCoords[:,2:4]+torch.unsqueeze(e3,dim=1)*barycentricCoords[:,4:]
        return p

    def vertexIdToCoordinates2D(self,results,vertexCoords):
        """Get cartesian coordinates of vertices in 2D.
            :param results: tensor of vertices
            :type results: torch.tensor
            :param vertexCoords: tensor of vertex coordinates
            :type vertexCoords: torch.tensor
            :return: cartesian coordinates
            :rtype: torch.tensor"""
        results=results[results>=0]
        coordinates=torch.zeros((results.size()[0],6),device=self.device)
        coordinates[:,0:2]=vertexCoords[self.simplices[results.long()][:,0].long()]
        coordinates[:,2:4]=vertexCoords[self.simplices[results.long()][:,1].long()]
        coordinates[:,4:]=vertexCoords[self.simplices[results.long()][:,2].long()]
        return coordinates

    def cartesianToBarycentric2D(self,nodeCoordinates, p):
        """Transformation from cartesian to barycentric coordinates in 2D.
            :param nodeCoordinates: cartesian coordinates of simplex nodes
            :type nodeCoordinates: torch.tensor
            :param p: cartesian coordiantes of point
            :type p: torch.tensor
            :return e1,e2,e3: barycentric coordinates
            :rtype: torch.tensor"""
        x1=nodeCoordinates[:,0]
        x2=nodeCoordinates[:,2]
        x3=nodeCoordinates[:,4]
        y1=nodeCoordinates[:,1]
        y2=nodeCoordinates[:,3]
        y3=nodeCoordinates[:,5]
        e1=((y2-y3)*(p[:,0]-x3)+(x3-x2)*(p[:,1]-y3))/((y2-y3)*(x1-x3)+(x3-x2)*(y1-y3))
        e2=((y3-y1)*(p[:,0]-x3)+(x1-x3)*(p[:,1]-y3))/((y2-y3)*(x1-x3)+(x3-x2)*(y1-y3))
        e3=1-e1-e2
        return e1,e2,e3

    def correctionForce2D(self,intersectingSimplices,nodesDeformed):
        """Calculates the intersection2D depth of selfintersecting points to the nearest surface in 2D.
            :param nodesDeformed: displaced points
            :type nodesDeformed: torch.tensor
            :param intersectingSimplices: (# number of deformed points, 1); list of simplices where deformed points lie in; -1 means the point is not inside any simplex;
            :type intersectingSimplices: torch.tensor
            :return: summed depth of selfintersecting points
            :rtype: torch.tensor"""
        sel1=intersectingSimplices>=0
        sel2=self.vertMask == 1
        sel=sel1*sel2
        points = nodesDeformed[sel]
        intersectingSimplicesNew = intersectingSimplices[sel]
        coordinatesDeformed = self.vertexIdToCoordinates2D(intersectingSimplicesNew, nodesDeformed)
        coordinatesOrig = self.vertexIdToCoordinates2D(intersectingSimplicesNew, self.verticesUndeformed)
        e1, e2, e3 = self.cartesianToBarycentric2D(coordinatesDeformed, points)
        resamp = self.barycentricToCartesian2D(coordinatesOrig, e1, e2, e3)
        inter_sdf, _, _ = self.intP(resamp, self.sdf)
        return torch.sum(inter_sdf**2, dim=1)




    def intersectionDetection2D(self,nodesDeformed, simplices):
        """Computes whether a point lies inside a simplex after displacement. Points are assumed to be nodes of simplices, so hits with their own simplex are ignored. In 2D.
            :param nodesDeformed: displaced points
            :type nodesDeformed: torch.tensor
            :param simplices: list of simplices containing node Ids
            :type simplices: torch.tensor
            :return: (# number of nodesDeformed, 1); list of simplices where deformed points lie in
            :rtype: torch.tensor"""
        vertex0=nodesDeformed[simplices[:,0].long(),:]
        vertex1=nodesDeformed[simplices[:,1].long(),:]-vertex0
        vertex2=nodesDeformed[simplices[:,2].long(),:]-vertex0
        numberOfTriangles=len(simplices)
        vertice_1_expand=vertex1.T.reshape((2,1,numberOfTriangles))
        vertice_2_expand=vertex2.T.reshape((2,1,numberOfTriangles))
        mat = torch.cat((vertice_1_expand,vertice_2_expand), dim=1)
        inv_mat = torch.linalg.inv(mat.T).T
        if nodesDeformed.size(0)==2:
            nodesDeformed=nodesDeformed.reshape((1,2))
        numberNodesDeformed=nodesDeformed.shape[0]
        ori_expand=torch.repeat_interleave(vertex0[:,:,None], numberNodesDeformed, dim=2)
        new_points=torch.einsum('imk,kmj->kij',inv_mat,nodesDeformed.T-ori_expand)
        values=torch.all(new_points>=0, dim=1) & torch.all(new_points <=1, dim=1) & (torch.sum(new_points, dim=1)<=1)
        return_values=torch.nonzero(values)
        id_simplices=return_values[:,0]
        simplices=simplices[id_simplices]
        id_p=return_values[:,1]
        id_simplices_new=id_simplices[torch.any(simplices==torch.unsqueeze(id_p,dim=1),dim=1)==False]
        id_p_new=id_p[torch.any(simplices==torch.unsqueeze(id_p,dim=1),dim=1)==False]
        result = -torch.ones(numberNodesDeformed, dtype=id_simplices.dtype,device=self.device)
        result[id_p_new]=id_simplices_new
        return result

    def batching2D(self,nodesDeformed):
        """ Computes selfintersecting points in batches in 2D.
            :param nodesDeformed: deformed points
            :type nodesDeformed: torch.tensor
            :return intersecting_points:  (# number of deformed points, 1); list of simplices where deformed points lie in
            :rtype intersecting_points: torch.tensor"""
        numberTriangles=torch.tensor(self.simplices.size(0),device=self.device)
        max_num=1000
        iter=int((numberTriangles/max_num).floor()+1)
        results=torch.zeros((iter,nodesDeformed.size(0)),device=self.device)
        for cnt in range(iter):
            if cnt==0:
                simplicesTemp=self.simplices[cnt*max_num:(cnt+1)*max_num]
                resTemp=self.intersectionDetection2D(nodesDeformed, simplicesTemp)
                resTemp=torch.where(resTemp>=0,resTemp+(cnt*max_num),resTemp)
            elif cnt==iter-1:
                simplicesTemp=self.simplices[cnt*max_num:]
                resTemp=self.intersectionDetection2D(nodesDeformed, simplicesTemp)
                resTemp=torch.where(resTemp>=0,resTemp+(cnt*max_num),resTemp)
            else:
                simplicesTemp=self.simplices[cnt*max_num:(cnt+1)*max_num]
                resTemp=self.intersectionDetection2D(nodesDeformed, simplicesTemp)
                resTemp=torch.where(resTemp>=0,resTemp+(cnt*max_num),resTemp)
            results[cnt]=resTemp
        intersecting_points=torch.max(results,dim=0)[0]
        return intersecting_points

    def loss2D(self,nodesDeformed):
        """Calculates the selfintersection loss in 2D.
            :param x: deformation parameters
            :type x: torch.tensor
            :return resulting_loss: summed depth of seflintersecting points
            :rtype resulting_loss: torch.tensor"""
        intersectingNodes=self.batching2D(nodesDeformed)
        sel1=intersectingNodes>=0
        sel2=self.vertMask == 1
        sel=sel1*sel2
        #if intersectingNodes[intersectingNodes >= 0].size(0) == 0:
        if intersectingNodes[sel].size(0) == 0:
            resulting_loss=torch.zeros(1).to(device=self.device)
        else:
            resulting_loss=self.correctionForce2D(intersectingNodes,nodesDeformed)
        return resulting_loss

    def currentIntersection2D(self,x):
        '''starts selfintersection algorithm'''
        nodesDeformed=self.dis(x)
        intersectingNodes=self.batching2D(nodesDeformed)
        return intersectingNodes
    def apply(self,nodesDeformed, simplices, coords):
        """Checks if any coordinates in coords are covered by 2 or more simplices.
            :param nodesDeformed: displaced points
            :type nodesDeformed: torch.tensor
            :param simplices: list of simplices containing node Ids
            :type simplices: torch.tensor
            :param coords: coordinates to be checked (#coords,3)
            :type coords: torch.tensor
            :return: tensor of coordinates covered by multiple simplices
            :rtype: torch.tensor"""
        vertex1 = nodesDeformed[simplices[:, 0].long(), :]
        vertex2 = nodesDeformed[simplices[:, 1].long(), :] - vertex1
        vertex3 = nodesDeformed[simplices[:, 2].long(), :] - vertex1
        numberOfTetrahedras = len(simplices)
        vertex2_expand = vertex2.T.reshape((2, 1, numberOfTetrahedras))
        vertex3_expand = vertex3.T.reshape((2, 1, numberOfTetrahedras))
        mat = torch.cat((vertex2_expand, vertex3_expand), dim=1)
        inv_mat = torch.linalg.inv(mat.T).T  #
        if coords.size(0) == 3:
            coords = coords.reshape((1, 3))
        numberCoords = coords.shape[0]
        vertex1_expand = torch.repeat_interleave(vertex1[:, :, None], numberCoords, dim=2)
        new_points = torch.einsum('imk,kmj->kij', inv_mat, coords.T - vertex1_expand)
        values = torch.all(new_points >= 0, dim=1) & torch.all(new_points <= 1, dim=1) & (
                    torch.sum(new_points, dim=1) <= 1)
        return_values = torch.nonzero(values)
        flattened = return_values[:, 1].flatten()
        unique_values, counts = torch.unique(flattened, return_counts=True)
        duplicate_values = unique_values[counts > 1]
        return coords[duplicate_values, :]







class selfintersectionDetection3D():
    """ Class for self-collision detection based on meshes.
    Implementation for point-tetrahedron check is taken from https://stackoverflow.com/questions/25179693/how-to-check-whether-the-point-is-in-the-tetrahedron-or-not.
    This self-collision detection implementation calculates the self-collisons at t=1 of deformation.
    Should be used as main term  in ADMM.
    :param pts: points used for collision detection
    :type pts: torch.tensor
    :param sdf: signed distance map of scene with object that should avoid self-intersection2D
    :type sdf: torch.tensor
    :param interpolator: CUDA device or cpu, see torch docs
    :type interpolator: interpolation class
    :param grads: gradients of sdf [x,y,z]
    :type grads: list of torch.tensors
    :param simplices: simplices of mesh (either triangles or tetrahedra); (#of simplices,3 or 4)
    :type simplices: torch.tensor
    :param vertices: vertices of mesh (# of vertices, 2 or 3)
    :type vertices: torch.tensor
    :param coef: coefficient applied to the collision loss
    :type coef: float
    :param device: sets the computation device, see torch
    :type device: string
    """
    def __init__(self,pts,sdf,interpolator,grads,simplices,verticesUndeformed,coef,device,pointsMask,pointsMaskLabel,unmaskedDis,vertMask):
        #pts isnodesOrig
        """constructor method"""
        self.sdf=sdf
        self.intP=interpolator
        self.grads=grads #has to have the following form (dimension,x,y,z)
        self.simplices=simplices
        self.verticesUndeformed=verticesUndeformed
        self.device=device
        self.coef=coef
        self.pointsMask=pointsMask
        self.pointsMaskLabel=pointsMaskLabel
        self.unmaskedDis=unmaskedDis
        self.vertMask=vertMask
        self.siPartner=None
        self.name=None
        self.pts=pts



    def __call__(self,dis=0,**kwargs):
        """Executing 3D self collision detection.
        :param dis: displacement of pts
        :type dis: torch.tensor
        :return: summed depth of selfintersecting points
        :rtype: torch.tensor"""
        if self.pointsMask is not None:
            nodesMoved=self.pts[self.pointsMask==self.pointsMaskLabel].clone()
            #nodesMoved=self.nodesOrig.clone()
            dis=dis[self.pointsMask==self.pointsMaskLabel]
            nodesMoved[self.vertMask==1]=nodesMoved[self.vertMask==1]+dis[self.vertMask==1]
            nodesMoved[self.vertMask==0]=nodesMoved[self.vertMask==0]+self.unmaskedDis
            self.siPartner.unmaskedDis=dis[self.vertMask==1].data

        else:
            nodesMoved=self.pts+dis
        return self.loss3D(nodesMoved)*self.coef



    def intersectionDetection3D(self,nodesDeformed, simplices):
        """Computes whether a point lies inside a simplex after displacement. Points are assumed to be nodes of simplices, so hits with their own simplex are ignored. In 3D.
            :param nodesDeformed: displaced points
            :type nodesDeformed: torch.tensor
            :param simplices: list of simplices containing node Ids
            :type simplices: torch.tensor
            :return: (# number of nodesDeformed, 1); list of simplices where deformed points lie in
            :rtype: torch.tensor"""
        vertex1 = nodesDeformed[simplices[:, 0].long(), :]
        vertex2 = nodesDeformed[simplices[:, 1].long(), :] - vertex1
        vertex3 = nodesDeformed[simplices[:, 2].long(), :] - vertex1
        vertex4 = nodesDeformed[simplices[:, 3].long(), :] - vertex1
        numberOfTetrahedras = len(simplices)
        vertex2_expand = vertex2.T.reshape((3, 1, numberOfTetrahedras))
        vertex3_expand = vertex3.T.reshape((3, 1, numberOfTetrahedras))
        vertex4_expand = vertex4.T.reshape((3, 1, numberOfTetrahedras))
        mat = torch.cat((vertex2_expand, vertex3_expand, vertex4_expand), dim=1)
        inv_mat = torch.linalg.inv(mat.T).T  #
        if nodesDeformed.size(0) == 3:
            nodesDeformed = nodesDeformed.reshape((1, 3))
        numberNodesDeformed = nodesDeformed.shape[0]
        vertex1_expand = torch.repeat_interleave(vertex1[:, :, None], numberNodesDeformed, dim=2)
        new_points = torch.einsum('imk,kmj->kij', inv_mat, nodesDeformed.T - vertex1_expand)
        values = torch.all(new_points >= 0, dim=1) & torch.all(new_points <= 1, dim=1) & (torch.sum(new_points, dim=1) <= 1)
        return_values = torch.nonzero(values)
        id_tet = return_values[:, 0]
        simplices = simplices[id_tet]
        id_p = return_values[:, 1]
        id_tet_new = id_tet[torch.any(simplices == torch.unsqueeze(id_p, dim=1), dim=1) == False]
        id_p_new = id_p[torch.any(simplices == torch.unsqueeze(id_p, dim=1), dim=1) == False]
        res = -torch.ones(numberNodesDeformed, dtype=id_tet.dtype, device=self.device)  # Sentinel value
        res[id_p_new] = id_tet_new
        return res

    def batching3D(self,nodesDeformed):
        """ Computes selfintersecting points in batches in 3D.
            :param nodesDeformed: deformed points
            :type nodesDeformed: torch.tensor
            :return intersecting_points:  (# number of deformed points, 1); list of simplices where deformed points lie in
            :rtype intersecting_points: torch.tensor"""
        numberTetrahedras = torch.tensor(self.simplices.size(0), device=self.device)
        max_num = 1000
        iter = int((numberTetrahedras / max_num).floor() + 1)
        results = torch.zeros((iter, nodesDeformed.size(0)), device=self.device)
        for cnt in range(iter):
            if cnt == 0:
                simplicesTemp = self.simplices[cnt * max_num:(cnt + 1) * max_num]
                resTemp = self.intersectionDetection3D(nodesDeformed, simplicesTemp)
                resTemp = torch.where(resTemp >= 0, resTemp + (cnt * max_num), resTemp)
            elif cnt == iter - 1:
                simplicesTemp = self.simplices[cnt * max_num:]
                resTemp = self.intersectionDetection3D(nodesDeformed, simplicesTemp)
                resTemp = torch.where(resTemp >= 0, resTemp + (cnt * max_num), resTemp)
            else:
                simplicesTemp = self.simplices[cnt * max_num:(cnt + 1) * max_num]
                resTemp = self.intersectionDetection3D(nodesDeformed, simplicesTemp)
                resTemp = torch.where(resTemp >= 0, resTemp + (cnt * max_num), resTemp)
            results[cnt] = resTemp
        res = torch.max(results, dim=0)[0]
        return res

    def barycentricToCartesian3D(self,nodeCoordinates, e1, e2, e3, e4):
        """Transformation from barycentric to cartesian coordinates in 3D.
            :param barycentricCoords: barycentric coordinates
            :type barycentricCoords: torch.tensor
            :param e1,e2,e3,e4: cartesian coordiantes of simpex
            :type e1,e2,e3,e4: torch.tensor
            :return: cartesian coordiantes
            :rtype: torch.tensor"""
        p = torch.unsqueeze(e1, dim=1) * nodeCoordinates[:, 0:3] + torch.unsqueeze(e2, dim=1) * nodeCoordinates[:,
                                                                                                 3:6] + torch.unsqueeze(
            e3, dim=1) * nodeCoordinates[:, 6:9] + torch.unsqueeze(e4, dim=1) * nodeCoordinates[:, 9:]
        return p

    def vertexIdToCoordinates3D(self,results, vertexCoords):
        """Get cartesian coordinates of vertices in 3D.
            :param results: tensor of vertices
            :type results: torch.tensor
            :param vertexCoords: tensor of vertex coordinates
            :type vertexCoords: torch.tensor
            :return: cartesian coordinates
            :rtype: torch.tensor"""
        results = results[results >= 0]
        coordinates = torch.zeros((results.size()[0], 12), device=self.device)
        coordinates[:, 0:3] = vertexCoords[self.simplices[results.long()][:, 0].long()]
        coordinates[:, 3:6] = vertexCoords[self.simplices[results.long()][:, 1].long()]
        coordinates[:, 6:9] = vertexCoords[self.simplices[results.long()][:, 2].long()]
        coordinates[:, 9:] = vertexCoords[self.simplices[results.long()][:, 3].long()]
        return coordinates

    def cartesianToBarycentric3D(self,nodeCoordinates, p):
        """Transformation from cartesian to barycentric coordinates in 3D.
            :param nodeCoordinates: cartesian coordinates of simplex nodes
            :type nodeCoordinates: torch.tensor
            :param p: cartesian coordiantes of point
            :type p: torch.tensor
            :return e1,e2,e3,e4: barycentric coordinates
            :rtype: torch.tensor"""
        a = nodeCoordinates[:, 0:3]
        b = nodeCoordinates[:, 3:6]
        c = nodeCoordinates[:, 6:9]
        d = nodeCoordinates[:, 9:]

        vbp = p - b
        vbd = d - b
        vbc = c - b
        vap = p - a
        vac = c - a
        vad = d - a
        vab = b - a

        va = (1 / 6) * torch.einsum('ij,ij->i', vbp.float(), torch.cross(vbd, vbc))
        vb = (1 / 6) * torch.einsum('ij,ij->i', vap.float(), torch.cross(vac, vad))
        vc = (1 / 6) * torch.einsum('ij,ij->i', vap.float(), torch.cross(vad, vab))
        vd = (1 / 6) * torch.einsum('ij,ij->i', vap.float(), torch.cross(vab, vac))
        v = (1 / 6) * torch.einsum('ij,ij->i', vab.float(), torch.cross(vac, vad))
        e1 = va / v
        e2 = vb / v
        e3 = vc / v
        e4 = vd / v

        return e1, e2, e3, e4

    def loss3D(self,nodesDeformed):
        """Calculates the selfintersection loss in 3D.
            :param x: deformation parameters
            :type x: torch.tensor
            :return resulting_loss: summed depth of seflintersecting points
            :rtype resulting_loss: torch.tensor"""

        intersectingNodes = self.batching3D(nodesDeformed)
        sel1=intersectingNodes>=0
        sel2=self.vertMask == 1
        sel=sel1*sel2
        #if intersectingNodes[intersectingNodes >= 0].size(0) == 0:
        if intersectingNodes[sel].size(0) == 0:
            res = torch.zeros(1).to(device=self.device)
        else:
            res = self.correctionForce3D(intersectingNodes, nodesDeformed)
        return res

    def correctionForce3D(self,intersectingNodes, nodesDeformed):
        """Calculates the intersection2D depth of selfintersecting points to the nearest surface in 3D.
            :param nodesDeformed: displaced points
            :type nodesDeformed: torch.tensor
            :param intersectingNodes: (# number of deformed points, 1); list of simplices where deformed points lie in; -1 means the point is not inside any simplex;
            :type intersectingNodes: torch.tensor
            :return: summed depth of selfintersecting points
            :rtype: torch.tensor"""
        sel1=intersectingNodes>=0
        sel2=self.vertMask == 1
        sel=sel1*sel2
        points = nodesDeformed[sel]
        intersectingNodes = intersectingNodes[sel]

        coordinatesDeformed = self.vertexIdToCoordinates3D(intersectingNodes,nodesDeformed)
        coordinatesOrig = self.vertexIdToCoordinates3D(intersectingNodes, self.verticesUndeformed)

        e1, e2, e3, e4 = self.cartesianToBarycentric3D(coordinatesDeformed, points)
        resamp = self.barycentricToCartesian3D(coordinatesOrig, e1, e2, e3, e4)
        inter_sdf,_,_  = self.intP (resamp, self.sdf)
        return torch.sum(inter_sdf**2)


    def currentIntersection3D(self,x):
        '''starts selfintersection algorithm'''
        nodesDeformed=self.dis(x)
        intersectingNodes=self.batching3D(nodesDeformed)
        return intersectingNodes

    def intersectionsNodes(self,nodesDeformed):
        """Calculates the selfintersection loss in 3D.
            :param x: deformation parameters
            :type x: torch.tensor
            :return resulting_loss: summed depth of seflintersecting points
            :rtype resulting_loss: torch.tensor"""
        intersectingNodes = self.batching3D(nodesDeformed)
        print("here",intersectingNodes.size())
        if intersectingNodes[intersectingNodes >= 0].size(0) == 0:
            res = torch.zeros(1).to(device=self.device)
        else:
            res = self.correctionForce3D(intersectingNodes, nodesDeformed)
        selector=intersectingNodes >= 0
        return res,nodesDeformed[intersectingNodes >= 0],selector,intersectingNodes[intersectingNodes >= 0]

