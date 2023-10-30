import os
import multiprocessing

def process_files(files_chunk):
    for file in files_chunk:
        index = int(file.split("_")[3])

        if index < 230:
            target_folder = os.path.join(directory, "train", particle_name)
        elif index < 240:
            target_folder = os.path.join(directory, "val", particle_name)
        else:
            target_folder = os.path.join(directory, "test", particle_name)

        os.rename(os.path.join(folder, file), os.path.join(target_folder, file))

def chunk_list(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

directory = '/pscratch/sd/r/rradev/larndsim_throws_converted' 

particle_names = ['electron', 'gamma', 'muon', 'pion', 'proton']
for particle_name in particle_names:
    folder = os.path.join(directory, particle_name)
    files = os.listdir(folder)

    # Pre-create directories
    for phase in ["train", "val", "test"]:
        target_folder = os.path.join(directory, phase, particle_name)
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

    # Create a pool of worker processes
    pool = multiprocessing.Pool(128)

    # Process files in parallel, in chunks
    chunk_size = 100  # adjust as needed
    pool.map(process_files, chunk_list(files, chunk_size))

    # Close the pool to release resources
    pool.close()
    pool.join()