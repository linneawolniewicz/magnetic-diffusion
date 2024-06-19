import numpy as np
import pandas as pd
import h5py
import torch
import os
from torch.utils.data import Dataset
from torchvision import transforms

DATA_PATH = "/home/linneamw/sadow_koastore/personal/linneamw/research/mag_diff/data"

# Originally were 4 directories, corresponding to the following twist parameters and following 100 frames with index ranges:
# 10058706, 2, (151, 250)
# 10058780, 1, (163, 262)
# 10058816, 0.5, (205, 304)
# 10058892, 0, (187, 286)
# They were renamed on April 11/2024 by Linnea to be their twist parameters and have steps 0-99
TWIST_VALS = [2, 1, 0.5, 0]

# Each file has 8 variables of interest: rho, temp, vx, vy, vz, bx, by, bz
RHO_MINMAX = {
    2: (0.203553111992523, 1.6653850962813834),
    1: (0.6924202774599341, 1.465420838985941),
    0.5: (0.782294723768042, 1.3475491225113108),
    0: (0.6831313782424506, 1.5174428882494002)
}

TEMP_MINMAX = {
    2: (0.43598743749967483, 4.552754489525326),
    1: (1.0494920889358552, 2.3804171892409314),
    0.5: (1.0619621453253771, 1.9709735074054366),
    0: (1.0096128534528106, 1.9686452524824187)
}

VX_MINMAX = {
    2: (-2.4524775999951895, 2.4525565410274934),
    1: (-1.2851308714164054, 1.284775222712393),
    0.5: (-0.6996226110730087, 0.7001486879613839),
    0: (-0.842414901622873, 0.9023085863326882)
}

VY_MINMAX = {
    2: (-2.68314147303075, 2.683071413389918),
    1: (-1.073689837368453, 1.073420796996045),
    0.5: (-1.0955346544409834, 1.1111729724936439),
    0: (-1.6828534392187218, 1.6827973694605476)
}

VZ_MINMAX = {
    2: (-2.4936612025211797, 0.31794510690755673),
    1: (-0.46144003248001325, 0.24536298360259146),
    0.5: (-0.3545083099875354, 0.10644461181826878),
    0: (-0.4631310223018058, 0.14937325429792772)
}

BX_MINMAX = {
    2: (-2.1859277959651395, 1.606480483654063),
    1: (-1.1916099109023819, 0.7562192077730857),
    0.5: (-0.9691827360376774, 0.7428448736874713),
    0: (-0.8232947521378738, 0.7915502315133682)
}

BY_MINMAX = {
    2: (-2.567970100575452, 1.2889119938724396),
    1: (-0.8635289639969939, 0.8268839888744896),
    0.5: (-1.0111190332178481, 0.6419334663454785),
    0: (-1.3452441959071522, 0.7248301056987723)
}

BZ_MINMAX = {
    2: (-2.3600934270365754, 2.359978995463844),
    1: (-0.7884068234550152, 0.7978370351352811),
    0.5: (-0.6593732662790239, 0.7197645399781375),
    0: (-0.7732625940291924, 0.8004346991521085)
}

GLOBAL_MINMAX = {
    "rho": (0.203553111992523, 1.6653850962813834),
    "temp": (0.43598743749967483, 4.552754489525326),
    "vx": (-2.4524775999951895, 2.4525565410274934),
    "vy": (-2.68314147303075, 2.683071413389918),
    "vz": (-2.4936612025211797, 0.31794510690755673),
    "bx": (-2.1859277959651395, 1.606480483654063),
    "by": (-2.567970100575452, 1.2889119938724396),
    "bz": (-2.3600934270365754, 2.359978995463844)
}

def read_vars(file):
    with h5py.File(file, 'r') as f:
        # Read variables of interest
        bxh5 = f['bfield']['bx'][()]
        byh5 = f['bfield']['by'][()]
        bzh5 = f['bfield']['bz'][()]
        vxh5 = f['velocity']['vx'][()]
        vyh5 = f['velocity']['vy'][()]
        vzh5 = f['velocity']['vz'][()]
        rhoh5 = f['fluid']['rho'][()]
        temph5 = f['fluid']['temperature'][()]

        time = f['time'][()]
        step = f['step'][()]

        # Average to cell center locations and drop the ghost cells
        rho = rhoh5[1:,1:]
        temp = temph5[1:,1:]
        vx = (vxh5[1:,1:] + vxh5[:-1,1:] + vxh5[1:,:-1] + vxh5[:-1,:-1] )/4
        vy = (vyh5[1:,1:] + vyh5[:-1,1:] + vyh5[1:,:-1] + vyh5[:-1,:-1] )/4
        vz = (vzh5[1:,1:] + vzh5[:-1,1:] + vzh5[1:,:-1] + vzh5[:-1,:-1] )/4
        bx = (bxh5[1:,:-1]+ bxh5[1:,1:])/2
        by = (byh5[:-1,1:]+ byh5[1:,1:])/2
        bz = bzh5[1:,1:]

    return rho, temp, vx, vy, vz, bx, by, bz, time, step

def normalize(data, minmax):
    min_val, max_val = minmax
    return (data - min_val) / (max_val - min_val)

def denormalize(data, minmax):
    min_val, max_val = minmax
    return data * (max_val - min_val) + min_val

def downsample(data, factor):
    return data[::factor, ::factor]

# Load data for a single twist value
class MagDiffDataset(Dataset):
    CONDITIONING_STEPS = 2

    def __init__(self, twist_val, transform=None):
        self.twist_val = twist_val
        self.transform = transform

    def load_data(self, idx):
        # Load data from file 
        file = os.path.join(DATA_PATH, f"{self.twist_val}_{idx}.h5")
        rho, temp, vx, vy, vz, bx, by, bz, time, step = read_vars(file)
        data = np.stack([rho, temp, vx, vy, vz, bx, by, bz], axis=-1)

        # Normalize data
        for j in range(8):
            data[:,j] = normalize(data[:,j], GLOBAL_MINMAX[list(GLOBAL_MINMAX.keys())[j]])

        # Downsample data from 8x1024x1024 to 8x256x256
        for j in range(8):
            data[:,j] = downsample(data[:,j], 4)

        return data
    
    def __len__(self):
        return len(self.data)

    # For a given index, load the image. If a transform is provided, apply it. 
    # Return the image, and conditional embedding (previous and next time steps)
    def __getitem__(self, idx):
        assert idx >= 1 and idx <= 98, "Index must be between 1 and 98"

        sample = torch.tensor(self.load_data(idx), dtype=torch.float32)
        prev_sample = torch.tensor(self.load_data(idx - 1), dtype=torch.float32)
        next_sample = torch.tensor(self.load_data(idx + 1), dtype=torch.float32)

        if self.transform:
            sample = self.transform(sample)
            prev_sample = self.transform(prev_sample)
            next_sample = self.transform(next_sample)

        # concatenate previous and next time step images
        conditional_embedding = torch.cat([prev_sample, next_sample], dim=-1)
        
        # TODO: Check with Yusuke that this is what imagen code expects
        # return (8x256x256, #TODOD figure this dimension out)
        return sample, conditional_embedding

