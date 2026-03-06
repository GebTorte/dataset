import torch

class S2DataSet(torch.utils.data.DataSet):
    """
    Load s2 patches
    - maybe add selector for cloudfree here?
    - generate GT cloud mask here or within loader? consider RAM
    - 
    """

    def __init__(self, path):
        pass

    def __len__(self):
        pass