import argparse
import sys
import torch
from PIL import Image
import os
import numpy as np
import umap
import pandas as pd
import matplotlib.pyplot as plt
import glob
import dask.dataframe as dd
from sklearn.cluster import KMeans
import geopandas as gpd

from extractFeatures import getFeatures
from utils_functions import alphanumeric_sort
from getShapefile import generate_shapefile_sequential, sort_shp_by_northwest
from getSubimages import extract_subimages

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_device', type=str, default="cpu",
                        help='cuda else cpu')

    parser.add_argument('--image_path', type=str, default='Ariege/data/Ariege_img/Ariege_norm_2024-08-22_00-00-00.tif',
                        help='Input image path')
    parser.add_argument('--subimage_folder_path', type=str, default='Ariege/data/Ariege_subimg',
                        help='Input subimages folder path')
    parser.add_argument('--image_embd_folder_path', type=str, default='Ariege/data/Ariege_embd',
                        help='Input embeddings folder path')
    parser.add_argument('--image_shapefile', type=str, default='Ariege/data/Ariege_shp/clustering_ariege.shp',
                        help='Input image shapefile path')
    
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    if args.type_device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else :
        device = torch.device("cpu")
        
    extract_subimages(args.image_path, args.subimage_folder_path)
    generate_shapefile_sequential(args.subimage_folder_path, args.image_shapefile)
    sort_shp_by_northwest(args.image_shapefile)

    getFeatures(args.subimage_folder_path, args.image_embd_folder_path, device)  # ou importe le modèle

    # Clustering phase
    files_tif = glob.glob(os.path.join(args.image_embd_folder_path, "*.csv"))
    sorted_filenames = sorted(files_tif, key=alphanumeric_sort)

    # Read each file into a Dask DataFrame
    dfs = [dd.read_csv(filename) for filename in sorted_filenames]

    # Concatenate the Dask DataFrames along axis 0 (vertical concatenation)
    combined_data = dd.concat(dfs, axis=0)

    reducer = umap.UMAP()
    embedding = reducer.fit_transform(combined_data.compute())

    # Visualiser le résultat
    plt.figure(figsize=(10, 8))
    plt.scatter(embedding[:, 0], embedding[:, 1], s=1, alpha=0.5)
    plt.title("UMAP Projection")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.show(block=False)  

    input("Push  ↵")  # Pause le script

    
    while True:
        try:
            n_clusters = int(input("Number of clusters (k): "))
            if n_clusters > 0:
                break
            else:
                print("Please enter an integer value.")
        except ValueError:
            print("Invalid. Please enter an integer value.")

   
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(combined_data.compute())

    output_shapefile = args.image_shapefile
    gdf = gpd.read_file(output_shapefile)
    gdf['labels'] = labels
    gdf.to_file(output_shapefile)




