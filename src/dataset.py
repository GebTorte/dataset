import torch
import torchvision.transforms.functional as TF
import torchvision.transforms.v2 as v2
import tifffile
import numpy as np

# use method from SatelliteCloudGenerator fork at https://github.com/GebTorte/SatelliteCloudGenerator
from SatelliteCloudGenerator.src.band_magnitudes import stat_mag_scaler
from SatelliteCloudGenerator.src.CloudSimulator import add_cloud_and_shadow


class TestS2TIFDataSet(torch.utils.data.Dataset):
    # test dataset whihc loads cloudSEN12 GT masks (high)
    def __init__(self, img_paths, seed:int=42):
        self.img_paths = img_paths
        self.seed = seed
        self.crop_size=256

        if self.seed:
            torch.manual_seed(seed) 

        
        self.toFloat32Transform = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self):
        img = tifffile.imread(self.img_paths[idx])

        X = TF.to_tensor(img)

        X = v2.RandomCrop(size=self.crop_size)(X)

        X = self.toFloat32Transform(X)

        X = X[1:13, ...]

        # last band (band 15) is gt
        y = X[14, ...]

        return X, y 




class TestS2TIFDataSet512(torch.utils.data.Dataset):
    # test dataset whihc loads cloudSEN12 GT masks (high)
    def __init__(self, img_paths, seed:int=42):
        self.img_paths = img_paths
        self.seed = seed
        self.crop_size=256

        if self.seed:
            torch.manual_seed(seed) 

        
        self.toFloat32Transform = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self):
        img = tifffile.imread(self.img_paths[idx])

        X = TF.to_tensor(img)

        X = v2.Pad([0,0,3,3], padding_mode="reflect")(X)

        X = self.toFloat32Transform(X)

        X = X[1:13, ...]

        # last band (band 15) is gt
        y = X[14, ...].long()

        return X, y 

    

class S2TIFDataSet(torch.utils.data.Dataset):
    """
    Generating Cloud GT ontop of cloudfree images.
    Load s2 (select bands 1:13) patches.
    Using RandomCrop to get 256x256 (crop_size) patches. 

    TODO
    - maybe add selector for cloudfree here? 
    """

    def __init__(self,
                img_paths,
                data_root,
                transparency_threshold:float = 0.05,
                seed:int|None=42, 
                randomness:float = 0.01,
                omitt_band_idxs:list[int] = [], # default omitt 10?
                crop_size:int=256,
                thick_cloud_percent: float = 0.7,
                thin_cloud_percent: float = 0.3,
                locality_degree: tuple[int, int] = (1,4),
        ):
        """
        
        seed: int - needed for reproducible synthetic cloud mask generation
        """
        self.img_paths = img_paths
        self.data_root = data_root
        self.seed = seed
        self.omitt_band_idxs = omitt_band_idxs # excluding cirrus by default
        self.crop_size = crop_size
        self.randomness = randomness
        self.transparency_threshold = transparency_threshold
        self.thick_cloud_percent = thick_cloud_percent
        self.thin_cloud_percent = thin_cloud_percent
        self.locality_degree = locality_degree

        # set torch random seed for random transform ops
        if self.seed:
            torch.manual_seed(seed) 

        self.transform_rc = v2.Compose([
            v2.RandomCrop(size=self.crop_size), # square
            #v2.ConvertImageDtype(torch.float32)
            #v2.ToDtype(torch.float32, scale=True),
        ])

        self.transforms = v2.Compose([
            v2.RandomCrop(size=self.crop_size),
            v2.RandomHorizontalFlip(p=0.5),
            #v2.ConvertImageDtype(torch.float32)
            #v2.ToDtype(torch.float32, scale=True),
        ])

        self.toFloat32Transform = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
        ])
        

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        """

        idx: int = index of the tensor to load
        """
        # apply transform/augmentation
        return self._get_randomcrop(idx=idx)

    
    def _get_randomcrop(self, idx):
        """

        idx: int = index of the tensor to load
        """
        img = tifffile.imread(self.img_paths[idx])

        X = TF.to_tensor(img)

        X = self.toFloat32Transform(X)

        X = self.transform_rc(X)

        # TODO: replace add_cloud_and_shadow... with CloudGenerator?
        # only pass bands idx 1-12, 
        # manual mask is at band 15 index 14,
        # band 14 is weird
        # band 1 index 0 is sentinel1 band  - use as feature perhaps, but not in cloudgenerator

        X = X[1:13, ...]

        shp = (1,12,256,256)
        dtype = float

        #cl1, cl2 = torch.zeros(*shp, dtype=dtype), torch.zeros(*shp, dtype=dtype)
        cl2 = None
        cmask, smask = torch.zeros(*shp, dtype=dtype), torch.zeros(*shp, dtype=dtype)
        cmask_thin, smask_thin = torch.zeros(*shp, dtype=dtype), torch.zeros(*shp, dtype=dtype)

        locality1 = torch.randint(*self.locality_degree, (1,)).item() if self.locality_degree[0] < self.locality_degree[-1] else 1

        # if torch.rand(1).item() < 1: # always go here # deprecate: self.thick_cloud_percent:
        cl1, cmask, smask = add_cloud_and_shadow(X,
            return_cloud=True,
            channel_magnitude=stat_mag_scaler(
                X,
                omitt_band_idxs=self.omitt_band_idxs, 
                seed=self.seed, 
                randomness=self.randomness
            ),
            cloud_color=True,
            locality_degree=locality1,
        )
        if torch.rand(1).item() < self.thin_cloud_percent:
            cl2, cmask_thin, smask_thin = add_cloud_and_shadow(cl1,
                return_cloud=True,
                channel_magnitude=stat_mag_scaler(
                    cl1,
                    omitt_band_idxs=self.omitt_band_idxs, 
                    seed=self.seed, 
                    randomness=self.randomness
                ),
                min_lvl=0.4,
                max_lvl=0.6,
                decay_factor=1.5,
                cloud_color=True,
                locality_degree=1,
            )

        # add cloud masks to cl
        # if no cloud generated, use clear image
        cl = cl2 if cl2 != None else cl1


        # convert transparency masks to binary masks
        # this is why we take < 1

        # Max across channels
        # < min_lvl from CloudSimulator
        max_cloud = torch.max(cmask, dim=1)[0]
        binary_mask_cloud = (max_cloud > self.transparency_threshold).long()

        max_cloud_thin = torch.max(cmask_thin, dim=1)[0]
        binary_mask_cloud_thin = (max_cloud_thin > self.transparency_threshold).long()

        # convert shadow masks
        #smask = torch.max(torch.clip(torch.add(smask, smask_thin), min=0, max=1), dim=1).values
        max_shadow = torch.clamp(smask+smask_thin, 0, 1).max(dim=1)[0]
        binary_mask_shadow = (max_shadow > self.transparency_threshold).long()

        # if non-exclusive masks create gt mask like this
        # because we work with quasi-refelctance and not probabilities
        # ranking cloud over thin cloud over shadow over clear 0
        y = torch.max(
            torch.stack([
                binary_mask_cloud * 3, 
                binary_mask_cloud_thin * 2, 
                binary_mask_shadow * 1,
            ]),
            dim=0
        )[0]
        
        # TODO add to other datasets
        # converting ranking-labels to cloudsen12-labels
        mapping = torch.tensor([0,3,2,1], device=X.device)
        y = mapping[y.squeeze().long()]

        # cl has to be (C, H, W)
        # y of shape (H,W)
        # squeeze as output from SatCloudGen has extra dimension
        return cl.squeeze(), y



class S2TIFDataSet512(torch.utils.data.Dataset):
    """
    Generating Cloud GT ontop of cloudfree images.
    Load s2 (select bands 1:13) patches.

    TODO
    - maybe add selector for cloudfree here? 
    """

    def __init__(self,
                img_paths,
                data_root,
                transparency_threshold:float = 0.05,
                seed:int|None=42, 
                randomness:float = 0.01,
                omitt_band_idxs:list[int] = [], # default omitt 10?
                crop_size:int=512,
                thick_cloud_percent: float = 0.7,
                thin_cloud_percent: float = 0.3,
                locality_degree: tuple[int, int] = (1,4),
        ):
        """
        
        seed: int - needed for reproducible synthetic cloud mask generation
        """
        self.img_paths = img_paths
        self.data_root = data_root
        self.seed = seed
        self.omitt_band_idxs = omitt_band_idxs # excluding cirrus by default
        self.crop_size = crop_size
        self.randomness = randomness
        self.transparency_threshold = transparency_threshold
        self.thick_cloud_percent = thick_cloud_percent
        self.thin_cloud_percent = thin_cloud_percent
        self.locality_degree = locality_degree

        # set torch random seed for random transform ops
        # only for this file!!
        if self.seed:
            torch.manual_seed(seed) 


        self.transforms = v2.Compose([
            v2.Pad([0,0,3,3], padding_mode="reflect"), # pad 509px to 512px
            v2.ToDtype(torch.float32, scale=True),
        ])

        self.augment = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
        ])

        self.toFloat32Transform = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
        ])
        

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        """

        idx: int = index of the tensor to load
        """
        # apply transform/augmentation
        return self._get_tensors(idx=idx)

    
    def _get_tensors(self, idx):
        """

        idx: int = index of the tensor to load
        """
        img = tifffile.imread(self.img_paths[idx])

        X = TF.to_tensor(img)

        X = self.transforms(X)

        X = self.augment(X)

        X = self.toFloat32Transform(X)


        # TODO: replace add_cloud_and_shadow... with CloudGenerator?
        # only pass bands idx 1-12, 
        # manual mask is at band 15 index 14,
        # band 14 is weird
        # band 1 index 0 is sentinel1 band  - use as feature perhaps, but not in cloudgenerator

        X = X[1:13, ...]

        shp = (1,12,self.crop_size,self.crop_size)
        dtype = float

        #cl1, cl2 = torch.zeros(*shp, dtype=dtype), torch.zeros(*shp, dtype=dtype)
        cl2 = None
        cmask, smask = torch.zeros(*shp, dtype=dtype), torch.zeros(*shp, dtype=dtype)
        cmask_thin, smask_thin = torch.zeros(*shp, dtype=dtype), torch.zeros(*shp, dtype=dtype)

        locality1 = torch.randint(*self.locality_degree, (1,)).item() if self.locality_degree[0] < self.locality_degree[-1] else 1

        # if torch.rand(1).item() < 1: # always go here # deprecate: self.thick_cloud_percent:
        cl1, cmask, smask = add_cloud_and_shadow(X,
            return_cloud=True,
            channel_magnitude=stat_mag_scaler(
                X,
                omitt_band_idxs=self.omitt_band_idxs, 
                seed=self.seed, 
                randomness=self.randomness
            ),
            cloud_color=True,
            locality_degree=locality1,
        )
        if torch.rand(1).item() < self.thin_cloud_percent:
            cl2, cmask_thin, smask_thin = add_cloud_and_shadow(cl1,
                return_cloud=True,
                channel_magnitude=stat_mag_scaler(
                    cl1,
                    omitt_band_idxs=self.omitt_band_idxs, 
                    seed=self.seed, 
                    randomness=self.randomness
                ),
                min_lvl=0.4,
                max_lvl=0.6,
                decay_factor=1.5,
                cloud_color=True,
                locality_degree=1,
            )

        # add cloud masks to cl
        # if no cloud generated, use clear image
        cl = cl2 if cl2 != None else cl1


        # convert transparency masks to binary masks
        # this is why we take < 1

        # Max across channels
        # < min_lvl from CloudSimulator
        max_cloud = torch.max(cmask, dim=1)[0]
        binary_mask_cloud = (max_cloud > self.transparency_threshold).long()

        max_cloud_thin = torch.max(cmask_thin, dim=1)[0]
        binary_mask_cloud_thin = (max_cloud_thin > self.transparency_threshold).long()

        # convert shadow masks
        #smask = torch.max(torch.clip(torch.add(smask, smask_thin), min=0, max=1), dim=1).values
        max_shadow = torch.clamp(smask+smask_thin, 0, 1).max(dim=1)[0]
        binary_mask_shadow = (max_shadow > self.transparency_threshold).long()

        # if non-exclusive masks create gt mask like this
        # because we work with quasi-refelctance and not probabilities
        # ranking cloud over thin cloud over shadow over clear 0
        y = torch.max(
            torch.stack([
                binary_mask_cloud * 3, 
                binary_mask_cloud_thin * 2, 
                binary_mask_shadow * 1,
            ]),
            dim=0
        )[0]
        
        # converting ranking-labels to cloudsen12-labels
        mapping = torch.tensor([0,3,2,1], device=X.device)
        y = mapping[y.squeeze().long()]

        # cl has to be (C, H, W)
        # y of shape (H,W)
        # squeeze as output from SatCloudGen has extra dimension
        return cl.squeeze(), y


class S2TIFDataSet_256_4x(torch.utils.data.Dataset):
    """
    Load s2 patches
    - maybe add selector for cloudfree here?
    - generate GT cloud mask here

    TODO (?): Adapt to 256x256 (patch images into 4 to use all training data)
    """

    def __init__(self, img_paths, data_root, transparency_threshold:float = 0.05, seed:int|None=42, randomness:float = 0.01, omitt_band_idxs:list[int] = [10]):
        """
        
        seed: int - needed for synthetic cloud mask generation
        """
        self.img_paths = img_paths
        self.data_root = data_root
        self.seed = seed
        self.omitt_band_idxs = omitt_band_idxs # excluding cirrus by default
        self.crop_size = crop_size
        self.randomness = randomness
        self.transparency_threshold = transparency_threshold

        # set torch random seed for random transform ops
        if self.seed:
            torch.manual_seed(seed) 

        self.transforms = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            #v2.ConvertImageDtype(torch.float32)
            #v2.ToDtype(torch.float32, scale=True),
        ])

        self.toFloat32Transform = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
        ])


    def __len__(self):
        return len(self.img_paths) * 4

    def __getitem__(self, idx):
        """

        idx: int = index of the tensor to load
        """
        # apply transform/augmentation
        return self._get_tensor(idx=idx)

    
    def _get_tensor(self, idx):
        """
        idx: int = index of the tensor to load
        """
        # 1. Map the 0 to (N*4)-1 index back to original data
        original_idx = idx // 4
        quad_idx = idx % 4  # 0=TL, 1=TR, 2=BL, 3=BR

        # loading creates 4x overhead!
        img = tifffile.imread(self.img_paths[original_idx])

        # only pass bands idx 0-11, manual mask is at band 15 index 14        
        
        X = TF.to_tensor(img)[1:13, ...]

        X = self.toFloat32Transform(X)

        # 2. Apply Reflect Padding (509 -> 512)
        # Pad: (left, right, top, bottom)
        X = F.pad(X, (0, 3, 0, 3), mode='reflect')

        # 3. Select the correct 256x256 quadrant
        if quad_idx == 0:   # Top-Left
            X = X[:, 0:256, 0:256]
        elif quad_idx == 1: # Top-Right
            X = X[:, 0:256, 256:512]
        elif quad_idx == 2: # Bottom-Left
            X = X[:, 256:512, 0:256]
        else:               # Bottom-Right
            X = X[:, 256:512, 256:512]

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

        # TODO: add info back to tensor and return this (as it includes S1 bands)
        #print(cl.shape, "cl shape")
        #X[3:, ...] = cl
        #X = X.unsqueeze(0)
    
        # convert transparency masks to binary masks
        # this is why we take < 1

        # Max across channels
        max_cloud = torch.max(cmask, dim=1)[0]
        # < min_lvl from CloudSimulator
        binary_mask_cloud = (max_cloud < self.transparency_threshold).long()

        max_shadow = torch.max(smask, dim=1)[0]
        binary_mask_shadow = (max_shadow < self.transparency_threshold).long()


        # We convert the masks to a single channel:
        # 0: no cloud, 1: cloud, 2: shadow
        # assuming exclusive masks
        #y = y[0] * 0 + y[1] * 1 + y[2] * 2


        # if non-exclusive masks create gt mask like this
        # TODO use argmax
        y_stacked = np.stack(np.zeros_like(binary_mask_cloud), binary_mask_cloud, binary_mask_shadow)
        y = np.argmax(y_stacked, axis=0)
        
        #y = torch.max(binary_mask_cloud * 2, binary_mask_shadow * 1) # ranking cloud over shadow

        # cl has to be (C, H, W)
        # y of shape (H,W)
        # squeeze as output from SatCloudGen has extra dimension
        return cl.squeeze(), y.squeeze()
