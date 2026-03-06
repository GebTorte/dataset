import torch
import torchvision.transform.functional as TF
import tifffile

# use method from SatelliteCloudGenerator fork at https://github.com/GebTorte/SatelliteCloudGenerator
from SatelliteCloudGenerator.src.band_magnitudes import stat_mag_scaler
from SatelliteCloudGenerator.src.CloudSimulator import add_cloud_and_shadow

class S2TIFDataSet(torch.utils.data.DataSet):
    """
    Load s2 patches
    - maybe add selector for cloudfree here?
    - generate GT cloud mask here
    - 


    NOTE: this loads images of dimension (b,512,512) from TIF 
    TODO (?): Adapt to 256x256 (patch images into 4?) 
    """

    def __init__(self, img_paths, seed:int=42, omitt_band_idxs:list[int] = [10]):
        """
        
        seed: int - needed for synthetic cloud mask generation
        """
        self.img_paths = img_paths
        self.seed = seed
        self.omitt_band_idxs = omitt_band_idxs # excluding cirrus by default

    def __len__(self):
        return len(self.img_paths)

    def __get__(self, idx, mode:str="edge"):
        """

        idx: int = index of the tensor to load
        
        mode: str = "edge", "symmetric", "reflect", "constant", and more padding modes supported by np.pad

        """
        # load TIF file from path at idx and return as Tensor

        img = tifffile.imread(self.img_paths[idx])

        X = TF.to_tensor(img)

        # add padding 3 in h and 3 in w direction 
        X = np.pad(X, ((0, 0), (1, 2), (1, 2)), mode, constant_values=0)

        # TODO: replace with CloudGenerator?
        cl, cmask, smask = add_cloud_and_shadow(X,
            return_cloud=True,
            channel_magnitude=stat_mag_scaler(
                torch.FloatTensor(X),
                bands=bands, 
                seed=seed, 
                randomness=0.01
            ),
            cloud_color=True,
        )

        # We convert the masks to a single channel:
        # 0: no cloud, 1: cloud, 2: shadow
        # assuming exclusive masks
        #y = y[0] * 0 + y[1] * 1 + y[2] * 2

        # otherwise this way
        y = np.max(cmask * 2, smask * 1) # ranking cloud over shadow

        return X, y
