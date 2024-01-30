# High-level API for loading particles
from .analysis_apis import data_reader, parse_particle

# TODO: particle event class with iterator, particle class

class particle_reader(data_reader):
    """
    A high level class specialized to read particle information
    """
    def __init__(self,*files):
        super(particle_reader, self).__init__()
        self.add_data('particle_mcst') # particle data
        for f in files:
            self.add_file(f)
            
    def get_event(self, n):
        """
        Get event from particle reader
        Args:
            n (int): index of event
        Return:
            particles (dict): particle information organized by key and associated array (index uniquely identify a particle)
        """
        self.read(n)
        return self.data('particle_mcst')


def get_particle(event, n):
    """
    Get individual particle from event
    Args:
        event (dict): particle information dict
        n (int): index of particle to look up
    Return:
        particle (dict): particle information organized by key
    """
    p = {}
    for k in event:
        p[k] = event[k][n]
    return p


def get_start(p):
    """
    Get particle start position
    input: particle dictionary
    output: x,y,z
    """
    return p['start_x'], p['start_y'], p['start_z']


def get_direction(p):
    """
    Get particle start direction
    input: particle dictionary
    output: px, py, pz
    """
    return p['direction_x'], p['direction_y'], p['direction_z']