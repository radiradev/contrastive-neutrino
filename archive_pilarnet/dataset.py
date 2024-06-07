
class CLRDataset(torchvision.datasets.DatasetFolder):
    def __init__(
            self,
            dataset_type,
            root='/pscratch/sd/r/rradev/converted_data/train',
            extensions='.npz',
            take_log = False,
            take_sqrt = True,
            clip = True,

    ):
        super().__init__(root=root, extensions=extensions, loader=self.loader)
        self.dataset_type = dataset_type
        self.take_log = take_log
        self.take_sqrt = take_sqrt
        self.clip = clip

    def loader(self, path):
        return np.load(path)
    
    def transform_energy(self, energy):
        if self.take_log:
            energy = torch.log(energy)
        if self.take_sqrt:
            energy = torch.sqrt(energy)
        if self.clip:
            energy = torch.clip(energy, -1, 1)
        return energy

    def preprocessing(self, sample):
        # split the energy and the coordinates
        coords, feat = np.split(sample['points'], [3], axis=1)

        # convert the label to an integer
        label = self.class_to_idx[str(sample['label'])]

        # convert to torch tensors
        coords = torch.from_numpy(coords).float()
        feat = torch.from_numpy(feat).float()
        label = torch.tensor(label).long()

        return coords, feat, label 
    
    def classifier_augmentations(self, coord, feat, label):
        funcs = np.random.choice([identity, drop, shift_energy, translate], 2)
        for func in funcs:
            coord, feat = func(coord, feat) 
        
        feat = self.transform_energy(feat)
        coord, feat = sparse_quantize(coord, feat)
        return coord, feat, label
    
    def contrastive_augmentations(self, coords_i, feat_i, label):
        coords_j, feat_j = coords_i.clone(), feat_i.clone()
        funcs = [rotate, drop, shift_energy, translate] 
        
        
        # draw functions and augment i
        funcs_i = np.random.choice(funcs, 2)
        funcs_j = np.random.choice(funcs, 2)

        for func in funcs_i:
            coords_i, feat_i = func(coords_i, feat_i)
        
        for func in funcs_j:
            coords_j, feat_j = func(coords_j, feat_j)

        feat_i = self.transform_energy(feat_i)
        feat_j = self.transform_energy(feat_j)
        coords_i, feat_i = sparse_quantize(coords_i, feat_i)
        coords_j, feat_j = sparse_quantize(coords_j, feat_j)

        return (coords_i, feat_i), (coords_j, feat_j) 
        


    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, _ = self.samples[index]
        sample = self.loader(path)
        coords, feat, label = self.preprocessing(sample)
        if self.dataset_type == 'contrastive':
            x_i, x_j = self.contrastive_augmentations(coords, feat, label)
            return x_i, x_j
        elif self.dataset_type == 'single_particle_augmented':
            coords, feat, label = self.classifier_augmentations(coords, feat, label)
            return coords, feat, label.unsqueeze(0)
        elif self.dataset_type == 'single_particle_base':
            feat = self.transform_energy(feat)
            coords, feat = sparse_quantize(coords, feat)
            return coords, feat, label.unsqueeze(0)
        else:
            raise ValueError(f'Unknown dataset type {self.dataset_type}')
        

if __name__ == '__main__':
    dataset = CLRDataset('contrastive')
    xi, xj = dataset[0]
    print(xi[0].shape, xj[0].shape)

    dataset_larnd = ThrowsDataset(dataset_type='single_particle_augmented', root='/pscratch/sd/r/rradev/larndsim_throws_converted_new/val')
    coords, feats, label = dataset_larnd[0]
    print(coords.shape, feats.shape, label.shape)