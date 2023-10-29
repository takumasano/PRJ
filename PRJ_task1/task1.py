import scipy.io
import matplotlib.pyplot as plt
# Load the .mat file (replace 'filename.mat' with your actual file name)
scipy.io.loadmat('/Users/tstakuma/Desktop/PRJ/samples/sample data/1_19-Oct-2019_10-18-23_mouse.mat')

def traj_withColour(x, y, fig=None, ax=None):
        if fig is None:
            fig, ax = plt.subplots()
        colors = np.linspace(0, 1, len(x))
        ax.plot(x, y, '-k', alpha=0.2)
        ax.scatter(x, y, c=colors, cmap='turbo')
        ax.plot(x[0], y[0], 'Dr', label='start', markersize=8)
        ax.axis('equal')
        norm = mpl.colors.Normalize(vmin=0, vmax=len(x))
        cbar = fig.colorbar(mpl.cm.ScalarMappable(
            cmap='turbo', norm=norm), ax=ax)
        cbar.set_label('Time step')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend()
        return fig, ax