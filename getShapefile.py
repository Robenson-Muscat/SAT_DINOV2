
import os
import geopandas as gpd
from shapely.geometry import Polygon
import numpy as np
import concurrent.futures
import functools
import warnings
import pandas as pd
import rioxarray
import warnings
import geopandas as gpd
from utils_functions import alphanumeric_sort


# Function to generate shapefile for a single image
def generate_shapefile(path, filename):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        file_path = os.path.join(path, filename)
        img = rioxarray.open_rasterio(file_path)
    bounds = img.rio.bounds()
    num_patches_x = img.rio.width // 16
    num_patches_y = img.rio.height // 16

    polys = []
    for j in range(num_patches_y):
        for i in range(num_patches_x):
            left = bounds[0] + i * 16 * img.rio.resolution()[0]
            top = bounds[3] - j * 16 * np.abs(img.rio.resolution()[1])
            right = left + 16 * img.rio.resolution()[0]
            bottom = top - 16 * np.abs(img.rio.resolution()[1])
            poly = Polygon([(left, top), (right, top), (right, bottom), (left, bottom), (left, top)])
            polys.append((poly, filename))
    return polys

# Function to generate shapefiles for multiple images sequentially
def generate_shapefile_sequential(path, output):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        filenames = [filename for filename in os.listdir(path) if filename.endswith('.tif')]
        sorted_filenames = sorted(filenames, key=alphanumeric_sort)
        
        polys_with_filenames = []
        for filename in sorted_filenames:
            try:
                polys_with_filenames.extend(generate_shapefile(path, filename))
            except Exception as exc:
                print(f'Error processing {filename}: {exc}')

        if not polys_with_filenames:
            raise ValueError("No valid geometries were processed.")

        polys, filenames = zip(*polys_with_filenames)
        gdf = gpd.GeoDataFrame({'geometry': polys, 'source_file': filenames})
        gdf.to_file(output)


#Sort by y-coordinate (north to south) then by x-coordinate (west to east)
def sort_by_northwest(gdf):
    return gdf.sort_values(by=['centroid_y', 'centroid_x'], ascending=[False, True])



def sort_shp_by_northwest(shp):
    #Sort the shapefile in the following order:
    #geographical position of the sub-image from which the patch is extracted, latitude(north to south), longitude(west to east)
    gp=gpd.read_file(shp)
    gp_sorted_fil = gp.sort_values(by='source_fil', key=lambda x: x.map(alphanumeric_sort))
    
    gp_sorted_fil['centroid'] = gp_sorted_fil.geometry.centroid
    gp_sorted_fil['centroid_x'] = gp_sorted_fil.centroid.x
    gp_sorted_fil['centroid_y'] = gp_sorted_fil.centroid.y
    
    
    # Apply the sort
    sorted_gdf_list = []
    for i in range(0, len(gp_sorted_fil['source_fil'].unique())):
        chunk = gp_sorted_fil.iloc[i*196:(i+1)*196]
        sorted_chunk = sort_by_northwest(chunk)
        sorted_gdf_list.append(sorted_chunk)
    
    
    gp_sorted = gpd.GeoDataFrame(pd.concat(sorted_gdf_list, ignore_index=True))
    gp_sorted=gp_sorted.drop(columns=['centroid_x','centroid_y','centroid'])

    gp_sorted.to_file(shp)
