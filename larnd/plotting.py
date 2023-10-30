import matplotlib.pyplot as plt

def plot_event(x, y, z, dE=None, event_id="", particle_name=""):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d', alpha=0.2)

    ax.scatter(x,y,z, c=dE, cmap='jet', s=10)
    ax.set_xlabel('x [cm]')
    ax.set_ylabel('y [cm]')
    ax.set_zlabel('z [cm]')
    ax.set_title(f'Event {event_id} ({particle_name})')


def plot_segments(segments_ev, event_id, particle_name):
    for segment in segments_ev:
            x_start, y_start, z_start = segment['x_start'], segment['y_start'], segment['z_start']
            x_end, y_end, z_end = segment['x_end'], segment['y_end'], segment['z_end']
            # in edep-sim axes are swapped
            ax.plot([z_start, z_end], [y_start, y_end], [x_start, x_end] , c='tab:gray', linewidth=2)
    
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d', alpha=0.2)
    ax.set_xlabel('x [cm]')
    ax.set_ylabel('y [cm]')
    ax.set_zlabel('z [cm]')
    ax.set_title(f'Event {event_id} ({particle_name})')
    