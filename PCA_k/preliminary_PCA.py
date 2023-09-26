import pandas as pd
from PCA_k.procrustes_utils import find_average
import numpy as np
from sklearn.decomposition import PCA

if __name__ == "__main__":
    obj_folder = '/media/mcgoug01/nvme/ThirdYear/kits23sncct_objdata/cleaned_objs'
    features_csv_fp = '/media/mcgoug01/nvme/ThirdYear/kits23sncct_objdata/features_labelled.csv'
    number_of_points = 1024
    n_iter = 20000
    tolerance = 1e-7
    n_components = 10

    df = pd.read_csv(features_csv_fp)

    # Select rows where all cyst and cancer measurements are 0
    columns_to_check = ['cyst_{}_vol'.format(i) for i in range(10)] + ['cancer_{}_vol'.format(i) for i in range(10)]
    df = df[df[columns_to_check].sum(axis=1) == 0]
    pca = PCA(n_components=n_components)
    # extract case and kidney position data, split between left and right
    df = df[['case', 'position']]
    df_left = df.loc[df['position'] == 'left']
    df_right = df.loc[df['position'] == 'right']
    data = []

    for df,side in zip([df_left,df_right],['left','right']):
        entry = {'position':side}
        average_pointcloud,aligned_pointclouds = find_average(df, obj_folder,number_of_points, n_iter, tolerance)
        aligned_shape = aligned_pointclouds.shape
        variance_point_clouds = np.array([pc - average_pointcloud for pc in aligned_pointclouds])
        variance_point_clouds = variance_point_clouds.reshape(variance_point_clouds.shape[0],-1).T
        cov_matrix = np.cov(variance_point_clouds)
        pca.fit(cov_matrix)
        entry['components'] = pca.components_.reshape((n_components,aligned_shape[1],aligned_shape[2]))
        entry['variance'] = pca.explained_variance_/np.sum(pca.explained_variance_)
        entry['average_pointcloud'] = average_pointcloud.reshape((aligned_shape[1],aligned_shape[2]))
        data.append(entry)