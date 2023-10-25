# High-level API for loading images
from .analysis_apis import data_reader, parse_sparse3d

class cluster_reader(data_reader):
    """
    A high level class specialized to obtain clusters form events
    """
    def __init__(self,*files):
        super(cluster_reader, self).__init__()
        self.add_data('sparse3d_group') # 3D image with cluster id in channel
        self.add_data('cluster3d_mcst') # clusters of voxels with energy in ID
        for f in files:
            self.add_file(f)
            
    def get_cluster_objects(self, n):
        """
        Args:
            n (int): index of event to read
        Return:
            clusters = array of (voxel, energy) images
                (voxels, energy)
                voxels: numpy array of size (N,3) of pixel coordinates
                energy: numpy vector of length N of energy 
        """
        # todo: use cluster3d_mcst data
        self.read(n)
        return self.data('cluster3d_mcst')

    def get_image(self, n):
        """
        Args:
            n (int): index of image to read
        Return:
            (voxels, energy, classes)
            voxels: numpy array of size (N,3) of pixel coordinates
            labels: numpy vector of length N of cluster labels
        """
        self.read(n)
        voxels, labels = parse_sparse3d(self.data('sparse3d_group')) # voxels, cluster labels
        return voxels, labels.flatten()