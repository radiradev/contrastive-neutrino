# High-level API for loading images
from .analysis_apis import data_reader, parse_sparse3d

class image_reader_3d(data_reader):
    """
    A high level class specialized to read from 3D images
    """
    def __init__(self, paths):
        super(image_reader_3d, self).__init__()
        self.add_data('sparse3d_data')      # voxel energy
        self.add_data('sparse3d_fivetypes') # voxel particle classification
        
        self.paths = paths if isinstance(paths, (list, tuple)) else [paths]
        for f in self.paths:
            self.add_file(f)
            
    def get_energy(self, n):
        """
        Args:
            n (int): index of image to read
        Return:
            (voxels, energy)
            voxels: numpy array of size (N,3) of pixel coordinates
            energy: numpy vector of length N of energy 
        """
        self.read(n)
        voxels, energy =  parse_sparse3d(self.data('sparse3d_data')) # voxels, energy
        return voxels, energy.flatten()

    def get_classes(self, n):
        """
        Args:
            n (int): index of image to read
        Return:
            (voxels, classes)
            voxels: numpy array of size (N,3) of pixel coordinates
            classes: numpy vector of length N of classes
        """
        self.read(n)
        voxels, classes = parse_sparse3d(self.data('sparse3d_fivetypes')) # voxels, classes
        return voxels, classes.flatten()

    def get_image(self, n):
        """
        Args:
            n (int): index of image to read
        Return:
            (voxels, energy, classes)
            voxels: numpy array of size (N,3) of pixel coordinates
            energy: numpy vector of length N of energy
            classes: numpy vector of length N of classes
        """
        self.read(n)
        voxels, energy = parse_sparse3d(self.data('sparse3d_data')) # voxels, energy
        _, classes = parse_sparse3d(self.data('sparse3d_fivetypes')) # voxels, classes
        return voxels, energy.flatten(), classes.flatten()