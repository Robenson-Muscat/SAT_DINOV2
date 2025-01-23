import os
import rasterio
from rasterio.windows import Window
from rasterio.transform import rowcol



def extract_subimages(filename, save_folder):

    #Size of sub-images
    subimage_size = 224


    os.makedirs(save_folder, exist_ok=True)


    with rasterio.open(filename) as src:
        width, height = src.width, src.height

        for y in range(0, height, subimage_size):
            for x in range(0, width, subimage_size):
                
                if x + subimage_size <= width and y + subimage_size <= height:
                    
                    window = Window(x, y, subimage_size, subimage_size)

                    # Read sub-image data
                    subimage = src.read(window=window)

                    
                    subimage_filename = f"subimage_{y}_{x}.tif"
                    subimage_path = os.path.join(save_folder, subimage_filename)

                    
                    with rasterio.open(
                        subimage_path,
                        "w",
                        driver="GTiff",
                        height=subimage_size,
                        width=subimage_size,
                        count=src.count,
                        dtype=src.dtypes[0],
                        crs=src.crs,
                        transform=rasterio.windows.transform(window, src.transform),
                    ) as dst:
                        dst.write(subimage)

    print(f"Extraction finished")
