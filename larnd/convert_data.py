import h5py
import numpy as np
from pathlib import Path
from LarpixParser import event_parser as EvtParser
from LarpixParser import hit_parser as HitParser
from LarpixParser import util as util
from multiprocessing import Pool, cpu_count
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm


common = Path('/global/cfs/cdirs/dune/users/rradev/contrastive')
larnd_path = Path(common /'larndsim_out')
files = sorted(list(larnd_path.glob('*h5')))

# module0, 2x2, 2x2_MR4, ndlar
detector = "ndlar"

run_config, geom_dict = util.detector_configuration(detector) 
output_folder = '/global/cfs/cdirs/dune/users/rradev/contrastive/npz_files'

def pdg_to_name(pdg):
    pdg_map = {11: 'electron', 22: 'gamma', 13: 'muon', -13: 'muon', 
               2212: 'proton', 211: 'pion', -211: 'pion'}
    return pdg_map[pdg]


def read_file(filename, num_expected_events=100):
    f = h5py.File(filename, 'r')
    packets = f['packets'] # readout
    vertices = f['vertices']
    trajectories = f['trajectories']
    assn = f['mc_packets_assn'] # association between readout and MC
    segments = f['tracks']

    pckt_event_ids = EvtParser.packet_to_eventid(assn, segments)
    
    trigger_mask = packets['packet_type'] == 7
    t0_grp = packets[trigger_mask].reshape(4, -1)['timestamp'][0] * run_config['CLOCK_CYCLE']
    event_ids = np.unique(pckt_event_ids[pckt_event_ids != -1]) 
    if len(t0_grp) != len(event_ids):
        raise ValueError(f'Number of events in t0 ({len(t0_grp)}) does not match number of events in event_ids ({len(event_ids)})')

    return packets, pckt_event_ids, trajectories, t0_grp, event_ids, vertices


def find_parent_particle(trajectories, event_id):
    traj_mask = trajectories['eventID'] == event_id
    traj_ev = trajectories[traj_mask]
    parent_particle = traj_ev[traj_ev['parentID'] == -1]

    return pdg_to_name(int(parent_particle['pdgId']))

        
def save_event(event_id, i_event, filename, packets, pckt_event_ids, trajectories, vertices, t0_grp, geom_dict, run_config):
    pckt_mask = pckt_event_ids == event_id
    packets_ev = packets[pckt_mask]
    t0 = t0_grp[i_event]
    vertices_ev = vertices[vertices['eventID'] == event_id]
    adc = np.array(packets_ev['dataword'])
    
    x, y, z, charge = HitParser.hit_parser_charge(t0, packets_ev, geom_dict, run_config)
    coordinates = np.stack(
        [np.array(x)/10, 
         np.array(y)/10, 
         np.array(z)/10]
        , axis=1)
    
    vertex = np.stack([vertices_ev['x_vert'], vertices_ev['y_vert'], vertices_ev['z_vert']]) / 10 # cm

    particle_class = find_parent_particle(trajectories, event_id)
    np.savez(f'{output_folder}/{particle_class}/{filename.stem}_eventID_{event_id}.npz', 
             adc=adc, 
             coordinates=coordinates,
             charge=charge,
             vertex=vertex
    )


def process_file(filename):
    try:
        packets, pckt_event_ids, trajectories, t0_grp, event_ids, vertices = read_file(filename)
        for i_event, event_id in enumerate(event_ids):
            save_event(event_id,i_event, filename, packets, pckt_event_ids, trajectories, vertices, t0_grp, geom_dict, run_config)
    except ValueError as e:
        print(f'Error in file {filename}: {e}')


if __name__ == '__main__':
    process_map(process_file, files, max_workers=cpu_count())