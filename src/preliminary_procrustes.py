import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import procrustes
import pandas as pd
import open3d as o3d

# Load the data
csv_fp = '/media/mcgoug01/nvme/ThirdYear/kits23sncct_objdata/features_labelled.csv'
df = pd.read_csv(csv_fp)

#kidneys are rows, measurements are columns in df
# df contains column on the first 10 cysts and 10 cancer volumes present in each kidney.
# we want to select kidneys with no cyst or cancer volumes present, i.e., when all 10 columns
# for cyst / cancers are = 0

# columns containing cyst measurements are labelled 'cyst_0_vol' to 'cyst_9_vol', and
# columns containing cancer measurements are labelled 'cancer_0_vol' to 'cancer_9_vol'

# select rows where all cyst and cancer measurements are 0
df = df.loc[(df['cyst_0_vol'] == 0) & (df['cyst_1_vol'] == 0) & (df['cyst_2_vol'] == 0) & (df['cyst_3_vol'] == 0) & (df['cyst_4_vol'] == 0) & (df['cyst_5_vol'] == 0) & (df['cyst_6_vol'] == 0) & (df['cyst_7_vol'] == 0) & (df['cyst_8_vol'] == 0) & (df['cyst_9_vol'] == 0) & (df['cancer_0_vol'] == 0) & (df['cancer_1_vol'] == 0) & (df['cancer_2_vol'] == 0) & (df['cancer_3_vol'] == 0) & (df['cancer_4_vol'] == 0) & (df['cancer_5_vol'] == 0) & (df['cancer_6_vol'] == 0) & (df['cancer_7_vol'] == 0) & (df['cancer_8_vol'] == 0) & (df['cancer_9_vol'] == 0)]

# now drop all columns except 'case', and 'position'
df = df[['case','position']]
# print the number of 'left' and 'right' kidneys, taken from the 'position' column
print(df['position'].value_counts())

# now we want to load the obj files of each kidney, and generate a pointcloud for each kidney
# we will then use procrustes analysis to align the pointclouds, and find the average kidney shape

# define the path to the obj files
obj_fp = '/media/mcgoug01/nvme/ThirdYear/kits23sncct_objdata/cleaned_objs'

# define a function to load the obj files
def load_obj(fp):
    """
    Load an obj file, and return a numpy array of the vertices
    """
    with open(fp) as f:
        lines = f.readlines()
    vertices = []
    for line in lines:
        if line.startswith('v'):
            vertex = line.split(' ')[1:]
            vertex = [float(v) for v in vertex]
            vertices.append(vertex)
    vertices = np.array(vertices)
    return vertices