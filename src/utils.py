import tifffile as tf 
import numpy as np
import os
from pathlib import Path

# functions for converting tiffs to pickle
# and deleting tiff on completion for saving space on disk

async def pickle_tiff(tif_path, npy_path):
    try:
        array = await tf.imread(path)
        np.save(npy_path, array)
        return True
    except Exception as e:
        print(f"Exception occured with file {tif_path}: {e}")
        return False

def delete_tiff(tif_path):
    os.remove(tif_path)

def convert_tiffs_en_masse(folder:Path=Path("./data/p509/high"), of:Path=Path("./data/p509/pickle"), delete_flag:bool=False, isuffix:str=".tif", osuffix:str=".npy"):
    subfolders = os.listdir(folder)

    if not os.path.exists(of):
        os.makedirs(of, exist_ok=True)

    for i, subfolder in enumerate(sorted(subfolders)):
        # TODO: collect jobs and spawn workers
        for file in os.listdir(subfolder):
            fpath = folder / subfolder / file
            npypath = folder / subfolder / (file.removesuffix() + osuffix)
            success = pickle_tiff(fpath, npypath)
            if delete_flag & success:
                delete_tiff(fpath)


if __name__ == "__main__":
    # TODO implement conversion for sephamore multiprocessing
    pass