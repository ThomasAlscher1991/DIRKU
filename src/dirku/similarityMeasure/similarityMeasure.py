import torch
class similarityMeasure():
    """ Template class for image similarity measures.
            :param pts: points to be displaced
            :type pts: torch.tensor
            :param val: image intensities at eval points in moving image
            :type val: torch.tensor
            :param data: reference or fixed image
            :type data: torch.tensor
            :param interpolatorIntensity: interpolation method used for image intensity interpolation in raster images.
            :type interpolatorIntensity: interpolation class
            :param coef: weight for similarity term
            :type coef: float
    """
    def __init__(self,pts,intensities,data,interpolatorIntensity,coef=1,pointsMask=None,pointsMaskLabel=None):
        """Constructor method"""
        self.intensities=intensities
        self.data=data
        self.coef=coef
        self.interpolatorIntensity=interpolatorIntensity
        self.pts=pts
        self.pointsMask=pointsMask
        self.pointsMaskLabel=pointsMaskLabel
    def measure(self,newIntensities,intensities):
        """Method populated by children"""
        pass
    def __call__(self,dis=0,**kwargs):
        """ Displaces point, feeds the measure method the coordinates of displaced points and returns the similarity between
            intensities at original coordinates vs displaced coordinates.
                :param dis: points displacements
                :type dis: torch.tensor
                :return: similiarity measure.
                :rtype: float
        """
        if self.pointsMask is not None:
            dis=dis[self.pointsMask==self.pointsMaskLabel]
            pts=self.pts[self.pointsMask==self.pointsMaskLabel]
            newIntensities, _, _ = self.interpolatorIntensity(pts + dis, self.data)

        else:
            newIntensities,_,_=self.interpolatorIntensity(self.pts+dis,self.data)
        return self.measure(newIntensities,self.intensities)*self.coef


class landmarkMatching():
    """ Child class for normalized cross correlation."""
    def __init__(self,pointsMoving, pointsFixed, ratio, coef=1):
        self.pointsMoving=pointsMoving
        self.pointsFixed=pointsFixed
        self.coef=coef
        self.ratio=ratio

    def __call__(self,dis=0,**kwargs):
        """ Displaces point, feeds the measure method the coordinates of displaced points and returns the similarity between
            intensities at original coordinates vs displaced coordinates.
                :param dis: points displacements
                :type dis: torch.tensor
                :return: similiarity measure.
                :rtype: float
        """
        pointsMoved=self.pointsMoving+dis
        diff=(self.pointsFixed-pointsMoved)*self.ratio
        return self.coef*(torch.sum(torch.norm(diff,dim=1)))



class ncc(similarityMeasure):
    """ Child class for normalized cross correlation."""
    def measure(self,newIntensities,intensities):
        """Computes normalized cross correlation.
            :param val_new: intensities at displaced points in fixed image
            :type val_new: torch.tensor
            :param val: intensities at original points in moving image
            :type val: torch.tensor
            :return: similiarity measure.
            :rtype: float
        """
        pair = torch.stack([newIntensities.flatten(), intensities.flatten()], dim=1)
        mx = torch.mean(pair, dim=0)
        temp=torch.mean((pair[:, 0] - mx[0]) * (pair[:, 1] - mx[1])) / (
           torch.prod(torch.std(pair, dim=0) + torch.finfo(torch.float32).eps))
        return 10-temp

class ssd(similarityMeasure):
    """ Child class for sum of squared differences.
    """
    def measure(self,newIntensities,intensities):
        """Computes sum of squared distances.
            :param val_new: intensities at displaced points in fixed image
            :type val_new: torch.tensor
            :param val: intensities at original points in moving image
            :type val: torch.tensor
            :return: similiarity measure.
            :rtype: float
        """
        pair = torch.stack([newIntensities.flatten(), intensities.flatten()], dim=1)
        mx = torch.sum((pair[:, 0] - pair[:, 1]) ** 2)
        return mx




class nmi(similarityMeasure):
    """ Child class for mutual information.
    """
    def Histogram2D(self, vals):
        """ Creates 3 histograms, 1 for the joint probability of intensity values and 2 for separate probabilities.
            intensities at original coordinates vs displaced coordinates.
                :param vals: combined intensities
                :type vals: torch.tensor
                :return hist: joint probability
                :rtype hist: torch.tensor
                :return hist_a: single probability
                :rtype hist_a: torch.tensor
                :return hist_b: single probability
                :rtype hist_b: torch.tensor
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

    def measure(self, newIntensities,intensities):
        """Computes normalized mutual information.
            :param val_new: intensities at displaced points in fixed image
            :type val_new: torch.tensor
            :param val: intensities at original points in moving image
            :type val: torch.tensor
            :return: similiarity measure.
            :rtype: float
        """
        self.device=newIntensities.device
        x = torch.stack([newIntensities.flatten(), intensities.flatten()], dim=1)
        h1, h2, h3 = self.Histogram2D(x)
        h1 = h1 / h1.sum()
        h2 = h2 / h2.sum()
        h3 = h3 / h3.sum()
        return 10 - ((torch.sum(-h2 * torch.log(h2)) + torch.sum(-h3 * torch.log(h3))) / torch.sum(-h1 * torch.log(h1)))

