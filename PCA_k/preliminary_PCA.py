import pandas as pd
from PCA_k.procrustes_utils import find_average
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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

    for df, side in zip([df_left,df_right],['left','right']):
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

        # plot the average point cloud and +/- 1 standard deviation of the first 2 components, where the central plot is just the average point cloud
        # in a 3x3 grid
        lim = np.abs(average_pointcloud).max()*1.1
        fig,ax = plt.subplots(1,3,subplot_kw={'projection':'3d'},figsize=(20,12))
        plt.subplots_adjust(wspace=0,hspace=0)

        for component_index in range(1,4):
            max_index = np.argsort(pca.explained_variance_)[-component_index]
            first_component = pca.components_[max_index].reshape((aligned_shape[1],aligned_shape[2]))*np.sqrt(pca.explained_variance_[max_index])*2

            ax[0].scatter(average_pointcloud[:,0]-first_component[:,0], average_pointcloud[:,1]-first_component[:,1],
                              average_pointcloud[:,2]-first_component[:,2])

            ax[1].scatter(average_pointcloud[:, 0], average_pointcloud[:, 1],
                               average_pointcloud[:, 2])

            ax[2].scatter(average_pointcloud[:,0]+first_component[:,0], average_pointcloud[:,1]+first_component[:,1],
                              average_pointcloud[:,2]+first_component[:,2])
            for j in range(3):
                ax[j].set_xlim(-lim, lim)
                ax[j].set_ylim(-lim, lim)
                ax[j].set_zlim(-lim, lim)

            fig.suptitle(side+' kidney, component '+str(component_index),fontsize=20)
            plt.show(block=True)