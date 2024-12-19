import torch
from typing import Optional, Type, Union, Tuple
from torch import Tensor
from ..interpolation import *
class similarityMeasure:
    """ Template class for image similarity measures.
    :param pts: points representing the deformable object
    :type pts: torch.Tensor
    :param intensities: intensities at material configuration
    :type intensities: torch.Tensor
    :param data: target intensity data
    :type data: torch.Tensor
    :param interpolatorIntensity: interpolator for pts locations after displacement in target intensity data
    :type interpolatorIntensity: nearest, linear, or cubic interpolation class
    :param coef: coefficient applied to the measure
    :type coef: float
    :param pointsMask: a mask for pts if only a subset of pts needs to be measured
    :type pointsMask: torch.Tensor
    :param pointsMaskLabel: the mask label for pointsMask that needs to be measured
    :type pointsMaskLabel: int
    """
    def __init__(self,pts: Tensor,intensities: Tensor,data: Tensor,interpolatorIntensity: Union[nearest,linear,cubic],coef: float=1,pointsMask: Optional[Tensor] = None,pointsMaskLabel: Optional[Tensor] = None):
        """Constructor method."""
        self.intensities=intensities
        self.data=data
        self.coef=coef
        self.interpolatorIntensity=interpolatorIntensity
        self.pts=pts
        self.pointsMask=pointsMask
        self.pointsMaskLabel=pointsMaskLabel
    def measure(self,newIntensities: Tensor,intensities: Tensor)->Tensor:
        """Method populated by children."""
        pass
    def __call__(self,dis: Tensor=0,**kwargs)->Tensor:
        """ Displaces point, feeds the measure method the coordinates of displaced points and returns the similarity between
            intensities at material coordinates vs spatial coordinates.
                :param dis: points displacements
                :type dis: torch.Tensor
                :return: similiarity measure
                :rtype: torch.Tensor
        """
        if self.pointsMask is not None:
            dis=dis[self.pointsMask==self.pointsMaskLabel]
            pts=self.pts[self.pointsMask==self.pointsMaskLabel]
            newIntensities, _, _ = self.interpolatorIntensity(pts + dis, self.data)

        else:
            newIntensities,_,_=self.interpolatorIntensity(self.pts+dis,self.data)
        return self.measure(newIntensities,self.intensities)*self.coef

class ncc(similarityMeasure):
    """ Child class for normalized cross correlation."""
    def measure(self,newIntensities: Tensor,intensities: Tensor)->Tensor:
        """Computes normalized cross correlation.
            :param newIntensities: intensities at spatial points in fixed image
            :type newIntensities: torch.Tensor
            :param val: intensities at material points in moving image
            :type val: torch.Tensor
            :return: similiarity measure.
            :rtype: torch.Tensor
        """
        pair = torch.stack([newIntensities.flatten(), intensities.flatten()], dim=1)
        mx = torch.mean(pair, dim=0)
        temp=torch.mean((pair[:, 0] - mx[0]) * (pair[:, 1] - mx[1])) / (
           torch.prod(torch.std(pair, dim=0) + torch.finfo(torch.float32).eps))
        return 10-temp

class ssd(similarityMeasure):
    """ Child class for sum of squared differences.
    """
    def measure(self,newIntensities: Tensor,intensities: Tensor)->Tensor:
        """Computes sum of squared differences.
            :param newIntensities: intensities at spatial points in fixed image
            :type newIntensities: torch.Tensor
            :param val: intensities at material points in moving image
            :type val: torch.Tensor
            :return: similiarity measure.
            :rtype: torch.Tensor
        """
        pair = torch.stack([newIntensities.flatten(), intensities.flatten()], dim=1)
        mx = torch.sum((pair[:, 0] - pair[:, 1]) ** 2)
        return mx




class nmi(similarityMeasure):
    """ Child class for mutual information.
    """
    def Histogram2D(self, vals: Tensor)->Tuple[Tensor,Tensor,Tensor]:
        """ Creates 3 histograms, 1 for the joint probability of intensity values and 2 for separate probabilities.
            intensities at original coordinates vs displaced coordinates.
                :param vals: combined intensities at material and spatial coordinates
                :type vals: torch.Tensor
                :return hist: joint probability
                :rtype hist: torch.Tensor
                :return hist_a: single probability
                :rtype hist_a: torch.Tensor
                :return hist_b: single probability
                :rtype hist_b: torch.Tensor
        """
        rangeh = torch.ceil(vals.max() - vals.min()).long()
        t_idx = vals.floor().long()
        p = torch.arange(vals.size(0))
        t = vals - t_idx
        ones4 = torch.ones([2, 2], dtype=torch.int32, device=self.device)
        onesp = torch.ones(t.size(0), dtype=torch.int32, device=self.device)
        stride_x, stride_y = torch.meshgrid(
            [torch.arange(0, 2, device=self.device) - 1, torch.arange(0, 2, device=self.device) - 1])
        t_idx = t_idx.flatten()
        indices = torch.einsum('a,bc->abc', t_idx[2 * p], ones4) * (rangeh)
        indices += torch.einsum('a,bc->abc', onesp, stride_x) * rangeh
        indices += torch.einsum('a,bc->abc', t_idx[2 * p + 1], ones4)
        indices += torch.einsum('a,bc->abc', onesp, stride_y)
        y = torch.stack([1 - t.flatten(), t.flatten()], dim=1)
        res = (torch.einsum('ab,ac->abc', y[2 * p, :], y[2 * p + 1, :]))
        v, ids = indices.flatten().unique(return_counts=True)
        val = torch.split(res.flatten(), ids.tolist());
        hist = torch.zeros(v.size(), device=self.device, dtype=torch.float32)
        va = (v % rangeh)
        vb = ((v / rangeh).long())
        for index, value in enumerate(val):
            hist[index] = value.sum()
        v_a, ids = va.unique(return_counts=True)
        hist_a = torch.zeros(v_a.size(), device=self.device, dtype=torch.float32)
        vala = torch.split(hist, ids.tolist());
        for index, value in enumerate(vala):
            hist_a[index] = value.sum()
        v_b, ids = vb.unique(return_counts=True)
        hist_b = torch.zeros(v_b.size(), device=self.device, dtype=torch.float32)
        valb = torch.split(hist, ids.tolist());
        for index, value in enumerate(valb):
            hist_b[index] = value.sum()
        hist = hist + torch.finfo(torch.float32).eps
        hist_a = hist_a + torch.finfo(torch.float32).eps
        hist_b = hist_b + torch.finfo(torch.float32).eps
        return hist, hist_a, hist_b

    def measure(self,newIntensities: Tensor,intensities: Tensor)->Tensor:
        """Computes normalized mutual information.
            :param newIntensities: intensities at spatial points in fixed image
            :type newIntensities: torch.Tensor
            :param val: intensities at material points in moving image
            :type val: torch.Tensor
            :return: similiarity measure.
            :rtype: torch.Tensor
        """
        self.device=newIntensities.device
        x = torch.stack([newIntensities.flatten(), intensities.flatten()], dim=1)
        h1, h2, h3 = self.Histogram2D(x)
        h1 = h1 / h1.sum()
        h2 = h2 / h2.sum()
        h3 = h3 / h3.sum()
        return 10 - ((torch.sum(-h2 * torch.log(h2)) + torch.sum(-h3 * torch.log(h3))) / torch.sum(-h1 * torch.log(h1)))

