from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    from ROOT import TChain
    from larcv import larcv
    import numpy as np
except ImportError:
    print('This API requires ROOT, larcv, and numpy installation!')
    import sys
    sys.exit(1)

def list_data(filepath):
    """
    A function to list data objects in larcv file
    Args:
        filepath (string): the full (or relative) path to a larcv binary data file
    Returns:
        list of object names
    """
    
    from ROOT import TFile
    f=TFile.Open(filepath)
    result = []
    for key in f.GetListOfKeys():
        name = str(key.GetName())
        if not name.endswith('_tree'): continue
        name = name.replace('_tree','')
        if name in result: continue
        result.append(name)
    f.Close()
    return result

class data_reader(object):
    """
    A class that can be useful to interface data files, move data index, and get data pointers.
    """
    def __init__(self):
        self._names = []       # a list of unique data strings in a file
        self._chains = []      # a list of ROOT TChain pointers (to be created)
        self._ptrs = {}        # a list of actual data pointers associated with each TChain data holder
        self._input_files = [] # a list of input data files (to be registered)

    def add_data(self,*names):
        """
        A function to add data to read.
        Args: 
            *names (list): a list of data identifier (unique) strings
        """
        for name in names:
            if name in self._names: continue
            self._names.append(name)
            self._chains.append(TChain(name + '_tree'))
            for f in self._input_files:
                if name in list_data(f):
                    self._chains[-1].AddFile(f)
                else:
                    print('Warning: %s is not in file %s' % (name,f))                    

    def add_file(self,*files):
        """
        A function to add files to read
        Args:
            *files (list): a list of strings to specify data file path
        """
        for f in files:
            if f in self._input_files: continue
            self._input_files.append(f)
            for c in self._chains:
                cname = str(c.GetName()).replace('_tree','')
                if cname in list_data(f):
                    c.AddFile(f)
                else:
                    print('Warning: %s is not in file %s' % (cname,f))

    def read(self,entry):
        """
        A function to read a specific entry in the input file(s)
        Args:
            entry (int): the entry (data index) number
        """
        for i,c in enumerate(self._chains):
            c.GetEntry(entry)
            self._ptrs[self._names[i]] = getattr(c,self._names[i] + '_branch')

    def data(self,name):
        """
        A function to retrieve a specific data instance pointer
        Args:
            name (string): the data identifer string given to add_data() function argument
        Return:
            data product pointer (a larcv C++ data product pointer)
        """
        return self._ptrs[name]

    def entry_count(self):
        """
        A function to retrieve the total number of entries from currently registered files.
        Return:
            num_entries (int): the total number of entries from currently registered files
        """
        if len(self._chains) < 1: return 0
        return self._chains[0].GetEntries()
    
    def __len__(self):
        return self.entry_count()
    

def parse_tensor2d(event_tensor2d):
    """
    A function to retrieve larcv::EventSparseTensor2D as a list of numpy arrays
    Args:
        event_tensor2d (larcv::EventSparseTensor2D): larcv C++ object for a collection of 2d sparse tensor objects
    Return:
        a python list of numpy arrays where each array represent one 2d tensor in dense matrix format
    """
    result = []
    for tensor2d in event_tensor2d.as_vector():
        img = larcv.as_image2d(tensor2d)
        result.append(np.array(larcv.as_ndarray(img)))
    return result

def parse_tensor3d(event_tensor3d):
    """
    A function to retrieve larcv::EventSparseTensor3D as a numpy array
    Args:
        event_tensor3d (larcv::EventSparseTensor3D): larcv C++ object for a 3d sparse tensor object
    Return:
        a numpy array of a dense 3d tensor object
    """
    return np.array(larcv.as_ndarray(event_tensor3d))

def parse_sparse2d(event_tensor2d):
    """
    A function to retrieve sparse tensor from larcv::EventSparseTensor2D object
    Args:
        event_tensor2d (larcv::EventSparseTensor2D): larcv C++ object for a collection of 2d sparse tensor objects
    Return:
        a python list of numpy array pair (coords,value) where coords has shape (N,3) 
            representing 3D pixel coordinate and value (N,1) stores pixel values.
    """
    result = []
    for tensor2d in event_tensor2d.as_vector():
        num_point = tensor2d.as_vector().size()
        np_voxels = np.zeros(shape=(num_point,2),dtype=np.int32)
        np_data   = np.zeros(shape=(num_point,1),dtype=np.float32)
        larcv.fill_2d_voxels(tensor2d,np_voxels)
        larcv.fill_2d_pcloud(tensor2d,np_data  )
        result.append((np_voxels,np_data))
    return result

def parse_sparse3d(event_tensor3d):
    """
    A function to retrieve sparse tensor from larcv::EventSparseTensor3D object
    Args:
        event_tensor3d (larcv::EventSparseTensor3D): larcv C++ object for a 3d sparse tensor object
    Return:
        a pair of numpy arrays (coords,value), where coords has shape (N,3) 
            representing 3D pixel coordinate and value (N,1) stores pixel values.
    """
    num_point = event_tensor3d.as_vector().size()
    np_voxels = np.zeros(shape=(num_point,3),dtype=np.int32)
    np_data   = np.zeros(shape=(num_point,1),dtype=np.float32)
    larcv.fill_3d_voxels(event_tensor3d,np_voxels)
    larcv.fill_3d_pcloud(event_tensor3d,np_data  )
    return np_voxels,np_data
        
def parse_particle(event_particle):
    """
    A function to parse larcv::EventParticle C++ object.
    Args:
        event_particle (larcv::EventParticle): a list of particles to be parsed
    Returns:
        dict: particle information organized by key and associated array (index uniquely idnetify a particle)
    """
    particle_v = event_particle.as_vector()
    num_particles = particle_v.size()
    part_info = {'particle_idx' : np.arange(0,num_particles),
                 'primary'      : np.zeros(num_particles,np.int8),
                 'pdg_code'     : np.zeros(num_particles,np.int32),
                 'mass'         : np.zeros(num_particles,np.float32),
                 'creation_x'   : np.zeros(num_particles,np.float32),
                 'creation_y'   : np.zeros(num_particles,np.float32),
                 'creation_z'   : np.zeros(num_particles,np.float32),
                 'direction_x'  : np.zeros(num_particles,np.float32),
                 'direction_y'  : np.zeros(num_particles,np.float32),
                 'direction_z'  : np.zeros(num_particles,np.float32),
                 'start_x'      : np.zeros(num_particles,np.float32),
                 'start_y'      : np.zeros(num_particles,np.float32),
                 'start_z'      : np.zeros(num_particles,np.float32),
                 'end_x'        : np.zeros(num_particles,np.float32),
                 'end_y'        : np.zeros(num_particles,np.float32),
                 'end_z'        : np.zeros(num_particles,np.float32),
                 'creation_energy'   : np.zeros(num_particles,np.float32),
                 'creation_momentum' : np.zeros(num_particles,np.float32),
                 'deposited_energy'  : np.zeros(num_particles,np.float32),
                 'npx'               : np.zeros(num_particles,np.int32),
                 'creation_process'  : ['']*num_particles,
                 'category'          : np.zeros(num_particles,np.int8),
                 'track_id' : np.zeros(num_particles,np.int32),
                 'parent_track_id' : np.zeros(num_particles,np.int32),
                 'num_voxels' : np.zeros(num_particles,np.int32)
                 }
    
    for idx in range(num_particles):
        particle = particle_v[idx]
        pdg_code = particle.pdg_code()
        mass     = larcv.ParticleMass(pdg_code)
        momentum = np.float32(np.sqrt(np.power(particle.px(),2)+
                                      np.power(particle.py(),2)+
                                      np.power(particle.pz(),2)))
        
        part_info[ 'primary'     ][idx] = np.int8(particle.track_id() == particle.parent_track_id())
        part_info[ 'pdg_code'    ][idx] = np.int32(pdg_code)
        part_info[ 'mass'        ][idx] = np.float32(mass)
        part_info[ 'creation_x'  ][idx] = np.float32(particle.x())
        part_info[ 'creation_y'  ][idx] = np.float32(particle.y())
        part_info[ 'creation_z'  ][idx] = np.float32(particle.z())
        part_info[ 'direction_x' ][idx] = np.float32(particle.px()/momentum)
        part_info[ 'direction_y' ][idx] = np.float32(particle.py()/momentum)
        part_info[ 'direction_z' ][idx] = np.float32(particle.pz()/momentum)
        part_info[ 'start_x'     ][idx] = np.float32(particle.first_step().x())
        part_info[ 'start_y'     ][idx] = np.float32(particle.first_step().y())
        part_info[ 'start_z'     ][idx] = np.float32(particle.first_step().z())
        part_info[ 'end_x'       ][idx] = np.float32(particle.last_step().x())
        part_info[ 'end_y'       ][idx] = np.float32(particle.last_step().y())
        part_info[ 'end_z'       ][idx] = np.float32(particle.last_step().z())
        part_info[ 'creation_energy'   ][idx] = np.float32(particle.energy_init() - mass)
        part_info[ 'creation_momentum' ][idx] = momentum
        part_info[ 'deposited_energy'  ][idx] = np.float32(particle.energy_deposit())
        part_info[ 'npx'               ][idx] = np.int32(particle.num_voxels())
        part_info[ 'parent_track_id'   ][idx] = np.int32(particle.parent_track_id())
        part_info[ 'track_id'   ][idx] = np.int32(particle.track_id())
        part_info[ 'num_voxels' ][idx] = np.int32(particle.num_voxels())

        category = -1
        process  = particle.creation_process()
        if(pdg_code == 2212 or pdg_code == -2212): category = 0
        elif not pdg_code in [11,-11,22]: category = 1
        elif pdg_code == 22: category = 2
        else:
            if process in ['primary','nCapture','conv','compt']: category = 2
            elif process in ['muIoni','hIoni']: category = 3
            elif process in ['muMinusCaptureAtRest','muPlusCaptureAtRest','Decay']: category = 4
            else:
                print('Unidentified process found: PDG=%d creation_process="%s"' % (pdg_code,process))
                raise ValueError
            
        part_info[ 'creation_process'  ][idx] = process
        part_info[ 'category'          ][idx] = category

    return part_info

class csv_writer:

    def __init__(self,fout):
        self._fout = fout
        self._keys = None

    def header(self,keys):
        if self._keys is None:
            self._fout = open(self._fout,'w')
            self._keys = list(keys)
            line = ''
            for k in keys:
                line += '{:s},'.format(k)
            self._fout.write(line.rstrip(',')+'\n')
            return True
        else:
            print('Cannot write the header twice...')
            return False

    def write(self,values):

        shape = np.shape(values)
        assert shape[0] == len(self._keys)
        
        for idx in range(shape[1]):
            line = ''
            for v in values:
                if type(v[idx]) == type(str()): line += '"{:s}",'.format(v[idx])
                else: line += '{:f},'.format(v[idx])
            self._fout.write(line.rstrip(',')+'\n')

    def flush(self):
        if self._keys is not None: self._fout.flush()

    def close(self):
        if self._keys is not None:
            self._fout.close()
