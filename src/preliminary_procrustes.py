import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
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

def process_dataframe(df):
    obj_folder = '/media/mcgoug01/nvme/ThirdYear/kits23sncct_objdata/cleaned_objs'
    pointclouds = []

    for index, row in df.iterrows():
        obj_fn = row['case'][:-7] + '_' + row['position'] + '.obj'
        obj_fp = os.path.join(obj_folder, obj_fn)
        vertices, simplices = load_obj_file(obj_fp)
        simplices = convert_rectangular_to_triangular(vertices, simplices)
        vertices = np.array(vertices)
        simplices = np.array(simplices)
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(simplices)
        pcd = mesh.sample_points_poisson_disk(number_of_points=1000)
        pointclouds.append(pcd)

    return pointclouds

def procrustes_analysis(target_points, reference_pointclouds, include_target=True):
    aligned_pointclouds = []

    target_cloud = o3d.geometry.PointCloud()
    target_cloud.points = o3d.utility.Vector3dVector(target_points)
    if include_target:
        aligned_pointclouds.append(np.asarray(target_cloud.points))

    icp_criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20000, relative_fitness=1e-7, relative_rmse=1e-7)

    for source_cloud in reference_pointclouds:
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source_cloud, target_cloud,
            max_correspondence_distance=1000,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=icp_criteria
        )
        source_cloud.transform(reg_p2p.transformation)
        pc = np.asarray(source_cloud.points)
        distances = cdist(target_cloud.points, pc)
        target_index, source_index = linear_sum_assignment(distances)
        aligned_pointclouds.append(np.asarray(source_cloud.points)[source_index])

    aligned_pointclouds = np.array(aligned_pointclouds)
    average_pointcloud = np.mean(aligned_pointclouds, axis=0)

    return average_pointcloud

# Main function
def main():
    csv_fp = '/media/mcgoug01/nvme/ThirdYear/kits23sncct_objdata/features_labelled.csv'
    df = pd.read_csv(csv_fp)

    # Select rows where all cyst and cancer measurements are 0
    columns_to_check = ['cyst_{}_vol'.format(i) for i in range(10)] + ['cancer_{}_vol'.format(i) for i in range(10)]
    df = df[df[columns_to_check].sum(axis=1) == 0]

    # Select only 'case' and 'position' columns
    df = df[['case', 'position']]
    print(df['position'].value_counts())

    df_left = df.loc[df['position'] == 'left']
    df_right = df.loc[df['position'] == 'right']
    for df in [df_left,df_right]:
        pointclouds = process_dataframe(df)
        target_pointcloud = np.asarray(pointclouds[0].points)
        target_pointcloud -= np.mean(target_pointcloud, axis=0)
        average_pointcloud = procrustes_analysis(target_pointcloud, pointclouds[1:],include_target=True)
        average_pointcloud -= np.mean(average_pointcloud, axis=0)
        average_pointcloud = procrustes_analysis(average_pointcloud,pointclouds,include_target=False)
        average_pointcloud -= np.mean(average_pointcloud, axis=0)

    # now we can plot the average pointcloud
    # plot the average pointcloud
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(average_pointcloud[:,0], average_pointcloud[:,1], average_pointcloud[:,2])
    lim = np.abs(average_pointcloud).max()
    ax.set_xlim(-lim,lim)
    ax.set_ylim(-lim,lim)
    ax.set_zlim(-lim,lim)

    plt.show(block=True)

if __name__ == "__main__":
    main()