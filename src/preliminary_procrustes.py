import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import procrustes, Delaunay
import pandas as pd
import open3d as o3d

#write function to read vertices from obj file
def load_obj_file(file_path):
    """
    Load an OBJ file and return vertices and faces.

    Parameters:
    - file_path: Path to the OBJ file

    Returns:
    - vertices: List of vertex coordinates
    - faces: List of faces represented as vertex indices
    """
    vertices = []
    faces = []

    try:
        with open(file_path, 'r') as obj_file:
            for line in obj_file:
                parts = line.strip().split()
                if not parts:
                    continue

                if parts[0] == 'v':
                    # Vertex definition
                    vertex = tuple(map(float, parts[1:]))
                    vertices.append(vertex)
                elif parts[0] == 'f':
                    # Face definition
                    face = [int(vertex.split('/')[0]) - 1 for vertex in parts[1:]]
                    if len(face) >= 3:
                        faces.append(face)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")

    return vertices, faces

def convert_rectangular_to_triangular(vertices, faces):
    """
    Convert a rectangular 3D mesh into a triangular 3D mesh by splitting all rectangular faces.

    Parameters:
    - vertices: List of vertex coordinates
    - faces: List of faces represented as vertex indices

    Returns:
    - new_faces: List of triangular faces
    """
    new_faces = []

    for face in faces:
        if len(face) == 4:
            # Split the rectangular face into two triangles
            triangle1 = [face[0], face[1], face[2]]
            triangle2 = [face[2], face[3], face[0]]
            new_faces.append(triangle1)
            new_faces.append(triangle2)
        else:
            # Keep existing triangular faces unchanged
            new_faces.append(face)

    return new_faces

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
obj_folder = '/media/mcgoug01/nvme/ThirdYear/kits23sncct_objdata/cleaned_objs'

#obj files are defined as case-'nii.gz'+ '_' + 'position' + '.obj'
# e.g. case_00000_left.obj
# use o3d to load the obj files and generate pointclouds

# create empty list to store pointclouds
pointclouds = []

# iterate through each row in df
for index, row in df.iterrows():
    # create the file path to the obj file
    obj_fn = row['case'][:-7] + '_' + row['position'] + '.obj'
    print(obj_folder, obj_fn)
    obj_fp = os.path.join(obj_folder, obj_fn)
    # load the obj file
    vertices,simplices = load_obj_file(obj_fp)
    # convert the rectangular mesh to a triangular mesh
    simplices = convert_rectangular_to_triangular(vertices, simplices)
    # convert the vertices and simplices to numpy arrays
    vertices = np.array(vertices)
    simplices = np.array(simplices)

    # load triangulation into open3d
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(simplices)
    # convert the mesh to a pointcloud
    pcd = mesh.sample_points_poisson_disk(number_of_points=10000)
    print(pcd)
    # add the pointcloud to the list of pointclouds
    pointclouds.append(pcd)

# now we have a list of pointclouds, we can use procrustes analysis to align them
# first we need to convert the pointclouds to numpy arrays
pointclouds = np.array(pointclouds)
print(pointclouds.shape)

# now we can use procrustes analysis to align the pointclouds
# we will use the first pointcloud as the reference pointcloud
# we will then align all other pointclouds to the reference pointcloud
# we will then average the aligned pointclouds to find the average kidney shape

# create empty list to store aligned pointclouds
aligned_pointclouds = []

# define reference pointcloud as first pointcloud
reference_pointcloud = pointclouds[0]

# iterate through each pointcloud
for pointcloud in pointclouds[1:]:
    # align the pointcloud to the reference pointcloud
    mtx1, mtx2, disparity = procrustes(reference_pointcloud, pointcloud)
    # add the aligned pointcloud to the list of aligned pointclouds
    aligned_pointclouds.append(mtx2)

# convert the list of aligned pointclouds to a numpy array
aligned_pointclouds = np.array(aligned_pointclouds)
print(aligned_pointclouds.shape)

# now we can average the aligned pointclouds to find the average kidney shape
# we will use the numpy mean function to average the aligned pointclouds
average_pointcloud = np.mean(aligned_pointclouds, axis=0)
print(average_pointcloud.shape)

# now we can plot the average kidney shape
# we will use matplotlib to plot the average kidney shape
# create a figure
fig = plt.figure()
# add a subplot
ax = fig.add_subplot(111, projection='3d')
# plot the average kidney shape
ax.scatter(average_pointcloud[:,0], average_pointcloud[:,1], average_pointcloud[:,2])
# set the axis limits
ax.set_xlim(-100,100)
ax.set_ylim(-100,100)
ax.set_zlim(-100,100)
# show the plot
plt.show()


