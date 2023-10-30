import h5py
import numpy as np
from pathlib import Path
from LarpixParser import event_parser as EvtParser
from LarpixParser import hit_parser as HitParser
from LarpixParser import util as util
from multiprocessing import Pool, cpu_count
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
import os

# module0, 2x2, 2x2_MR4, ndlar
detector = "ndlar"
common = Path('/global/cfs/cdirs/dune/users/rradev/contrastive/individual_particles')
larnd_path = Path(common /'larndsim_throws')

run_config, geom_dict = util.detector_configuration(detector)  
output_path = os.path.join(os.environ['PSCRATCH'], 'larndsim_throws_converted_new')

def pdg_to_name(pdg):
    pdg_map = {11: 'electron', 22: 'gamma', 13: 'muon', -13: 'muon', 
               2212: 'proton', 211: 'pion', -211: 'pion'}
    return pdg_map[pdg]


def read_file(filename, num_expected_events=100):
    f = h5py.File(filename, 'r')
    #if unable to read file, skip
    packets = f['packets'] # readout
    vertices = f['vertices']
    assn = f['mc_packets_assn'] # association between readout and MC
    segments = f['tracks']

    pckt_event_ids = EvtParser.packet_to_eventid(assn, segments)
    event_ids = np.unique(pckt_event_ids[pckt_event_ids != -1]) 


    trigger_mask = packets['packet_type'] == 7
    
    # I used reshape incorrectly here, it should be (n, 4) not (4, n)
    t0_grp = packets[trigger_mask].reshape(-1, 4)['timestamp'][:, 0] * run_config['CLOCK_CYCLE'] # fixed potential bug here 
    if len(t0_grp) != len(event_ids):
        # removing second repeated trigger
        unique_triggers, trigger_counts = np.unique(t0_grp, return_counts=True)
        repeated_triggers = unique_triggers[trigger_counts > 1]
        assert len(repeated_triggers) == 1, 'More than one repeated trigger found'
        # find the indices of the repeated triggers in t0_grp
        repeated_trigger_indices = np.where(np.isin(t0_grp, repeated_triggers))[0]
        assert len(repeated_trigger_indices) == 2, 'More than one pair of repeated triggers found'
        # remove the second trigger in the pair
        t0_grp = np.delete(t0_grp, repeated_trigger_indices[1]) # remove the second trigger

    assert len(t0_grp) == len(event_ids), f'Number of events in t0 ({len(t0_grp)}) does not match number of events in event_ids ({len(event_ids)})'

    return packets, pckt_event_ids, t0_grp, event_ids, vertices


def find_parent_particle(trajectories, event_id):
    traj_mask = trajectories['eventID'] == event_id
    traj_ev = trajectories[traj_mask]
    parent_particle = traj_ev[traj_ev['parentID'] == -1]

    return pdg_to_name(int(parent_particle['pdgId']))

        
def save_event(event_id, i_event, filename, packets, pckt_event_ids, vertices, t0_grp, geom_dict, run_config):
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

    particle_name = filename.stem.split('_')[0]
    assert particle_name in ['electron', 'gamma', 'muon', 'pion', 'proton']

    file_index = int(filename.stem.split("_")[3])
    
    if file_index < 230:
        target_folder = os.path.join(output_path, "train", particle_name)

    elif file_index < 240:
        target_folder = os.path.join(output_path, "val", particle_name)

    else:
        target_folder = os.path.join(output_path, "test", particle_name)
    
    if len(coordinates) > 3:
        np.savez(f'{target_folder}/{filename.stem}_eventID_{event_id}.npz', 
                adc=adc, 
                coordinates=coordinates,
                charge=charge,
                vertex=vertex
        )


def process_file(filename):
    try:
        packets, pckt_event_ids, t0_grp, event_ids, vertices = read_file(filename)
        for i_event, event_id in enumerate(event_ids):
            save_event(event_id,i_event, filename, packets, pckt_event_ids, vertices, t0_grp, geom_dict, run_config)
    except ValueError as e:
        pass 
        # print(f'Error in file {filename}: {e}')
    except KeyError as e:
        print(f'Key error in file {filename}: {e}')


if __name__ == '__main__':

    for particle_type in ['electron', 'muon', 'gamma', 'pion', 'proton']:
        print(f'Processing {particle_type} files')
        files = list(larnd_path.glob(f'{particle_type}*h5'))
        process_map(process_file, files, max_workers=128)