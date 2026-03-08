import torch
import torchvision.transforms.functional as TF
import torchvision.transforms.v2 as v2
import tifffile # TODO: load with PIL instead?

# use method from SatelliteCloudGenerator fork at https://github.com/GebTorte/SatelliteCloudGenerator
from SatelliteCloudGenerator.src.band_magnitudes import stat_mag_scaler
from SatelliteCloudGenerator.src.CloudSimulator import add_cloud_and_shadow

class S2TIFDataSet(torch.utils.data.Dataset):
    """
    Load s2 patches
    - maybe add selector for cloudfree here?
    - generate GT cloud mask here
    - 


    NOTE: this loads images of dimension (b,512,512) from TIF 
    TODO (?): Adapt to 256x256 (patch images into 4? or use Randomcrop!) 
    """

    def __init__(self, img_paths, data_root, seed:int=42, randomness:float = 0.01, omitt_band_idxs:list[int] = [10], crop_size:int=256):
        """
        
        seed: int - needed for synthetic cloud mask generation
        """
        self.img_paths = img_paths
        self.data_root = data_root
        self.seed = seed
        self.omitt_band_idxs = omitt_band_idxs # excluding cirrus by default
        self.crop_size = crop_size
        self.randomness=randomness

        # set torch random seed for random transform ops
        torch.manual_seed(seed) 

        self.transform_rc = v2.Compose([
            v2.RandomCrop(size=self.crop_size), # square
            v2.ConvertImageDtype(torch.float32)
            #v2.ToDtype(torch.float32, scale=True),
        ])

        self.transforms = v2.Compose([
            v2.RandomCrop(size=self.crop_size),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ConvertImageDtype(torch.float32)
            #v2.ToDtype(torch.float32, scale=True),
        ])
        

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx, mode:str="edge"):
        """

        idx: int = index of the tensor to load
        
        mode: str = "edge", "symmetric", "reflect", "constant", and more padding modes supported by np.pad

        """
        # apply transform/augmentation
        return self._get_randomcrop(idx=idx)

    
    def _get_randomcrop(self, idx):
        """

        idx: int = index of the tensor to load
        """
        # load TIF file from path at idx and return as Tensor
        # TODO: check if dimensions in right order
        img = tifffile.imread(self.img_paths[idx])

        X = TF.to_tensor(img)

        X = self.transform_rc(X)

        # TODO: replace with CloudGenerator?
        cl, cmask, smask = add_cloud_and_shadow(X,
            return_cloud=True,
            channel_magnitude=stat_mag_scaler(
                X,
                omitt_band_idxs=self.omitt_band_idxs, 
                seed=self.seed, 
                randomness=self.randomness
            ),
            cloud_color=True,
        )

        # We convert the masks to a single channel:
        # 0: no cloud, 1: cloud, 2: shadow
        # assuming exclusive masks
        #y = y[0] * 0 + y[1] * 1 + y[2] * 2

        # if non-exclusive masks transform like this
        y = np.argmax(cmask * 2, smask * 1) # ranking cloud over shadow

        return X, y


    def _get_transforms(self, idx):
        """

        idx: int = index of the tensor to load
        """
        # load TIF file from path at idx and return as Tensor

        img = tifffile.imread(self.img_paths[idx])

        X = TF.to_tensor(img)

        X = self.transforms(X)

        # TODO: replace with CloudGenerator?
        cl, cmask, smask = add_cloud_and_shadow(X,
            return_cloud=True,
            channel_magnitude=stat_mag_scaler(
                X,
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

        # if non-exclusive masks transform like this
        y = np.argmax(cmask * 2, smask * 1) # ranking cloud over shadow

        return X, y

    def _get_with_buffer(self, idx, mode:str="edge"):
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

