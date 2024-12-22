import numpy as np
import os
import re
import matplotlib.pyplot as plt
import json
from .postprocessing_utils import *
from typing import Optional, Type, Union, Tuple
from torch import Tensor

def visual_convergence(workingDirectory: str,segmentsOfInterest: Optional[list]=None):
    """ Plots convergence.
        :param workingDirectory: path to working directory, see docs
        :type workingDirectory: string
         :param segmentations: segmentations of interest list
         :type segmentations: list

    """
    if os.path.exists(os.path.join(workingDirectory,"results", "transformation_affine.json")):
        with open(os.path.join(workingDirectory, "results", "transformation_affine.json")) as json_file:
            data = json.load(json_file)
        fig,ax=plt.subplots(len(data.keys()))
        for i,key in enumerate(data.keys()):
            ax[i].plot(np.array(data[key]))
            ax[i].set_title(key)
        plt.suptitle('affine')
        plt.show()
    else:
        pass
    if segmentsOfInterest is not None:
        for s in segmentsOfInterest:
            print(s)
            if os.path.exists(os.path.join(workingDirectory,"results", f"transformation_affine_{s}.json")):
                with open(os.path.join(workingDirectory, "results", f"transformation_affine_{s}.json")) as json_file:
                    data = json.load(json_file)
                fig, ax = plt.subplots(len(data.keys()))
                for i, key in enumerate(data.keys()):
                    ax[i].plot(np.array(data[key]))
                    ax[i].set_title(key)
                plt.suptitle(f'affine segment {s}')
                plt.show()

    files = os.listdir(os.path.join(workingDirectory, "results"))
    filtered_files = [file for file in files if file.startswith("transformation_nonrigid") and file.endswith(".json")]

    if len(filtered_files) > 0:
        sorted_files = sorted(filtered_files,
                              key=lambda s: (extract_segment_and_scale_JSON(s)[0], extract_segment_and_scale_JSON(s)[1]),
                              reverse=True)
        for f in sorted_files:
            scale,segment=extract_segment_and_scale_JSON(f)
            with open(os.path.join(workingDirectory, "results", f)) as json_file:
                data = json.load(json_file)
            fig, ax = plt.subplots(len(data.keys()))
            for i, key in enumerate(data.keys()):
                ax[i].plot(data[key])
                ax[i].set_title(key)
            plt.suptitle(f'non rigid; segment {segment}; scale {scale}')
            plt.show()






