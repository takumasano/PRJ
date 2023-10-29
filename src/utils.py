# %%
import numpy as np
import pandas as pd
import glob
import re
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import PIL
import torch
from einops import rearrange, reduce, repeat
from torch.utils.data import TensorDataset, DataLoader
from alive_progress import alive_bar
from sklearn.preprocessing import StandardScaler, scale


# ---------------------------------------------------------------------------- #
#                                     info                                     #
# ---------------------------------------------------------------------------- #

# path
path_project = Path(__file__).parent.parent
path_data = path_project / 'data'
path_data_raw = [path_data/'raw'/'Keio Results',
                 path_data/'raw'/'Komagino Results']


def expon(x, a):
    return a * np.exp(-a*x)


def labels2idx(labels, minLen=3):
    minCount = 3
    unique, counts = np.unique(labels, return_counts=True)
    seqInfor = dict(zip(unique, counts))
    seq = {}
    for k, v in seqInfor.items():
        if v >= minLen:
            seq[k] = np.where(labels == k)[0]
    return seq


def obj2type(x, astype=np.float64):
    try:
        return x.astype(astype)
    except:
        return [obj2type(i, astype) for i in x]


def findMax(x):
    ''' find the max value and its index '''
    max_ = np.max(x)
    max_idx = np.unravel_index(np.argmax(x, axis=None), x.shape)
    return max_, max_idx


class ExpInfo:
    bad_subj = ['K-Reg-H-1', 'K-Reg-H-2', 'K-Reg-S-5']
    taskName = ['one_dot', 'three_dot', 'reaching']
    traj_columns_motor = ["x-shift", "y-shift"]
    traj_columns_disp = [['dot-x', 'dot-y'],
                         ['dot-x1', 'dot-y1', 'dot-x2',
                             'dot-y2', 'dot-x3', 'dot-y3'],
                         ['dot-x', 'dot-y']]
    screenSize = np.array((1900, 1060))

    @staticmethod
    def getScreenSise(df):
        if 'dot-x1' in df.columns:
            screenSize = df.loc[:, 'dot-x1':'dot-y3'].max().max()
        else:
            screenSize = df.loc[:, 'dot-x':'dot-y'].max().max()
        return screenSize

    @staticmethod
    def getSubjIDs():
        files = []
        for datapath in path_data_raw:
            files += glob.glob(str(datapath) + '/*')

        ids = []
        for file in files:
            id = re.search(r'((K-Reg)|(Reg))-(S|H)-\d+', file)
            if id is not None:
                ids.append(id.group())
        ids = set(ids).difference(ExpInfo.bad_subj)
        ids = list(ids)
        ids.sort()
        return ids

    @staticmethod
    def getSubjIDs_byGroup():
        ids = ExpInfo.getSubjIDs()
        id_H = []
        id_S = []
        for id in ids:
            if 'H' in id:
                id_H.append(id)
            else:
                id_S.append(id)
        return id_H, id_S


class LoadData:
    def __init__(self) -> None:
        pass

    @staticmethod
    def mouseMovement(subjID, task, trialno=None):
        fname = f'{subjID}_{task}.csv'
        fpath = path_data / 'Preprocessing' / 'mouseMovement' / fname
        df = pd.read_csv(fpath)
        if trialno is not None:
            try:
                df = df.loc[df['trialno'].isin(trialno)]
            except:
                try:
                    df = df.loc[df['trialno'] == trialno]
                except:
                    raise ValueError('trialno is not valid')
        df = df.loc[df["trialno"] != 0]
        return df

    @staticmethod
    def mouseMovement_array(subj, task, velocity=False, packDot=False):
        ''' return array of mouse movement: ([[trial_1], [trial_2]], [[trial_1], [trial_2]])
        '''
        df = LoadData.mouseMovement(subj, task)
        screenSize = ExpInfo.getScreenSise(df)
        trials = set(df['trialno'])
        xy = []
        xy_disp = []
        for trial in trials:
            df_ = df.query(f'trialno == {trial}').copy()
            xy_ = df_[["x-shift", "y-shift"]].values / screenSize

            if (task == 'one_dot') or (task == 'reaching'):
                xy_disp_ = df_[['dot-x', 'dot-y']].values / screenSize
            elif task == 'three_dot':
                xy_disp_ = df_[['dot-x1', 'dot-y1', 'dot-x2',
                                'dot-y2', 'dot-x3', 'dot-y3']].values / screenSize

            if velocity:
                xy_ = xy_[:-1, :]
                xy_disp_ = np.diff(xy_disp_, axis=0)

            xy.append(xy_)
            
            if (task == 'three_dot') and packDot:
                xy_disp_ = [xy_disp_[:, 0:2], xy_disp_[:, 2:4], xy_disp_[:, 4:6]]
            xy_disp.append(xy_disp_)

        return xy, xy_disp

    @staticmethod
    def mouseMovementRollingData(subjID='K-Reg-S-18', task='one_dot', wSize=48, interval=1, pos=False, nTrial_val=6, seed=0):
        # load data
        df = LoadData.mouseMovement(subjID, task)

        # Split data into train and test
        trial_train, trials_val = DataProcessing.split_train_val_trials(
            df, nTrial_val=nTrial_val, seed=seed)
        df_train = df.query(f'trialno in @trial_train')
        df_val = df.query(f'trialno in @trials_val')

        # rolling
        d_train = DataProcessing.rollingWindow_from_df(
            df_train, wSize, interval, pos=pos)
        d_val = DataProcessing.rollingWindow_from_df(
            df_val, wSize, interval, pos=pos)

        class TrajDataset(torch.utils.data.Dataset):
            def __init__(self, d):
                self.d = d

            def __len__(self):
                return self.d.shape[0]

            def __getitem__(self, idx):
                return self.d[idx]

        dataset_train = TrajDataset(d_train)
        dataset_val = TrajDataset(d_val)
        return dataset_train, dataset_val

    @staticmethod
    def behaviorData(subjID, task):
        files = []
        for datapath in path_data_raw:
            files += list(datapath.glob('*.*'))

        for file in files:
            if file.match(f'*{subjID}_*{task}_results.csv'):
                df = pd.read_csv(file, index_col=False)
                df['participant'] = df['participant'].str.strip()
                if 'H' in subjID:
                    df['group'] = 'H'
                else:
                    df['group'] = 'S'
                return df

    @staticmethod
    def xhy(subj, task, wSize=60, path='TrajNet_xhy'):
        filepath = path_data / path / f'{subj}_{task}_xhy_{wSize}.npz'
        d = np.load(filepath, allow_pickle=True)
        x, h, y = d['x'], d['h'], d['y']
        x = obj2type(x)
        h = obj2type(h)
        y = obj2type(y)
        return x, h, y

    @staticmethod
    def xhy_disp(subj, task, wSize=60, path='TrajNet_xhy'):
        filepath = path_data / path / f'{subj}_{task}_xhy_disp_{wSize}.npz'
        d = np.load(filepath, allow_pickle=True)
        x, h, y = d['x'], d['h'], d['y']
        x, h, y = obj2type(x), obj2type(h), obj2type(y)
        return x, h, y


class DataProcessing:

    @staticmethod
    def seqSegmentation(seq, dist_threshold, minLen=3):
        from sklearn.cluster import AgglomerativeClustering
        connectivity = np.diagflat(np.ones(len(seq)-1), 1)
        labels = AgglomerativeClustering(n_clusters=None,
                                         distance_threshold=dist_threshold,
                                         connectivity=connectivity,
                                         linkage='average').fit_predict(seq)
        return labels2idx(labels, minLen=minLen)

    @staticmethod
    def diff(x, measure='euclidean', offset=1):
        from scipy.spatial import distance
        dist = distance.pdist(x, measure)
        dist = distance.squareform(dist)
        dist = np.diagonal(dist, offset=offset)
        return dist

    @staticmethod
    def split_train_val_trials(df, nTrial_val=6, seed=0):
        trials = set(df['trialno']).difference([0])
        nTrial = len(trials)
        """_summary_

        Returns:
            _type_: _description_
        """        
        rng = np.random.default_rng(seed)
        trials_val = rng.choice(nTrial, nTrial_val, replace=False)
        trial_train = trials.difference(trials_val)
        return trial_train, trials_val

    @staticmethod
    def rollingWindow(d, wSize=60, interval=1, pos=False):
        """ Rolling window function along time dim
        Args:
            d (np.array): time x feature
            wSize (int): window size
            interval (int): interval size

        Returns:
            np.array: rolling windowed array
        """
        if d.shape[0] < wSize:
            raise ValueError('Data length is shorter than window size')
        d_ = []
        S = 0
        E = S + wSize
        while E <= (len(d)-1):
            d_.append(d[S:E, :])
            S += interval
            E = S + wSize
        d_ = np.stack(d_, axis=0)
        if pos:
            d_ = d_.cumsum(axis=1)
        return d_

    @staticmethod
    def rollingWindow_from_df(df, wSize, interval=1, pos=False, returnWithTrial=False):
        ''' Run rolling window function based on trial number
        '''
        screensize = ExpInfo.getScreenSise(df)
        d = []
        trials = list(set(df['trialno']).difference([0]))
        for trial in trials:
            df_ = df.query(f'trialno == {trial}').copy()
            df_ = df_[["x-shift", "y-shift"]].values / screensize
            d.append(DataProcessing.rollingWindow(
                df_, wSize, interval, pos=pos))

        if returnWithTrial:
            return d
        else:
            return np.concatenate(d, axis=0)

    @staticmethod
    def cart2pol(x, y):
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return(rho, phi)

    @staticmethod
    def positionEncoding_sincos_mat(nTime, dim=4, max_length=300):
        ''' Position encoding matrix '''
        x = np.arange(nTime) * 2 * np.pi / max_length
        x = np.tile(x, (dim, 1)).T  # t f
        x = x * np.arange(1, dim+1)
        x = np.hstack((np.sin(x), np.cos(x)))
        return x

    @staticmethod
    def seqTrim(x, minTime):
        ''' Trim the sequence to the minimum timeshift'''
        # x: b t f
        tLen = np.random.randint(minTime, x.shape[1])
        sTime = np.random.randint(0, x.shape[1]-tLen)
        eTime = sTime + tLen
        return x[:, sTime:eTime, :]

    @staticmethod
    def standardise_list(xList):
        ''' Standardise list of arrays'''
        xList_ = np.concatenate(xList, axis=0)
        scale = StandardScaler().fit(xList_)
        return [scale.transform(x) for x in xList]


class SynthData:
    def __init__(self) -> None:
        pass

    @staticmethod
    def spiral(nTime=72, nBatch=64, seed=0, add_polar=False):
        rng = np.random.default_rng(seed)
        XY = []
        for i in range(nBatch):
            theta = np.linspace(rng.uniform(0.5, 2*np.pi*4),
                                rng.uniform(0.5, 2*np.pi*4), nTime)
            r = np.linspace(rng.uniform(), 1, nTime)
            transform = rng.random((2, 2))

            def polar2z(r, theta):
                compx = r * np.exp(1j * theta)
                xy = np.vstack([np.real(compx), np.imag(compx)]).T
                return xy

            xy = polar2z(r, theta)
            if rng.random() > 0.5:
                r = -r

            xy = polar2z(r, theta)
            if rng.random() > 0.5:
                xy[:, 0] = -xy[:, 0]

            if rng.random() > 0.5:
                xy[:, 1] = -xy[:, 1]

            xy = np.roll(xy, 1, axis=1)
            xy = np.matmul(xy, transform)

            xy = xy / np.max(np.abs(xy))
            XY.append(xy)
        XY = np.stack(XY, 2)
        XY = np.transpose(XY, (2, 0, 1))

        if add_polar:
            x_, y_ = DataProcessing.cart2pol(XY[:, :, 0], XY[:, :, 1])
            x_ = repeat(x_, 'b t -> b t f', f=1)
            y_ = repeat(y_, 'b t -> b t f', f=1)
            XY = np.concatenate([XY, x_, y_], axis=2)
        return XY

    @staticmethod
    def spiral_dataset(**kwargs):
        x = SynthData.spiral(**kwargs)
        return TensorDataset(torch.from_numpy(x))

    @staticmethod
    def sin(nTime=72, nBatch=8, plot=False):
        data = [np.linspace(0, 2*np.pi, nTime),
                np.linspace(np.pi/2, np.pi/2+2*np.pi, nTime)]
        data = np.vstack(data).T
        data = np.sin(data)
        data = np.tile(data, (nBatch, 1, 1))
        data = data + np.random.random(data.shape)/2
        if plot:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1)
            ax.plot(data[0, :, 1])
            ax.plot(data[0, :, 0])
        return data
    
    @staticmethod
    def genReachingSeq(nReach=10, time=5, fs=60, velocity=True, seed=0):
        ''' Generate a sequence of reaching motion '''
        # reproducibility 
        rng = np.random.default_rng(seed)
        
        # --------------------------- locate target points --------------------------- #
        screenSize = ExpInfo.screenSize / (ExpInfo.screenSize.max())
        toCentre = screenSize[1]/2 * (4/3)
        
        # define 4 target points
        targetsLocation = np.array([[-toCentre, 0], [toCentre, 0], [0, -toCentre], [0, toCentre]])
        
        # --------------------------- random select targets -------------------------- #
        targetSet = np.arange(4)
        iTarget = [rng.choice(targetSet)]
        for i in range(1, nReach):
            targetSet_ = targetSet
            targetSet_ = np.delete(targetSet_, iTarget[-1])
            iTarget.append(rng.choice(targetSet_, 1))
        iTarget = np.hstack(iTarget)
        
        # ----------------------- interpolate reaching sequence ---------------------- #
        tp = time * fs # number of samples
        target = np.vstack([[0, 0], targetsLocation[iTarget]])
        itp = np.linspace(0, tp, nReach+1).astype(int)
        x = np.interp(np.arange(tp), itp, target[:, 0])
        y = np.interp(np.arange(tp), itp, target[:, 1])
        if velocity:
            x = np.diff(x)
            y = np.diff(y)
        xy = np.vstack([x, y]).T
        return xy, target
    
    @staticmethod
    def genReachingSeq_trial(nTrial, seed=0, **kwargs):
        xy = []
        target = []
        for i in range(nTrial):
            xy_, target_ = SynthData.genReachingSeq(seed=i+seed, **kwargs)
            xy.append(xy_)
            target.append(target_)
        return xy, target
            


class Plot:
    palette_group = ['#A96FBD','#6F89BD']

    @staticmethod
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

    @staticmethod
    def traj_withWeight(x, y, w, align='e', ax=None, seqColormap='viridis', minSize=10, maxSize=200):
        ''' Plot trajectory with weights
        align: 'e'(default) end, 's' start, 'c' center
        '''
        from sklearn.preprocessing import minmax_scale
        w = minmax_scale(w, feature_range=(minSize, maxSize))
        n = len(x)
        nW = len(w)
        if align == 's':
            offset = 0
        elif align == 'e':
            offset = n - nW
        elif align == 'c':
            offset = (n - nW)//2
        else:
            raise ValueError('align must be e, s, or c')

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        # plot line
        ax.plot(x, y, 'k', alpha=0.3)

        # plot sample points with color
        cmap = mpl.cm.get_cmap(seqColormap)
        colors = cmap(range(n))
        # ax.scatter(x, y, c=colors, s=minSize)
        ax.scatter(x, y, c='k', s=3, alpha=0.5)

        # plot starting point
        ax.plot(x[0], y[0], 'dr')

        # plot weights
        sc = ax.scatter(x[offset:offset+nW], y[offset:offset+nW],
                        c=colors[offset:offset+nW, :],
                        s=w,
                        edgecolors='k',
                        alpha=0.8)
        ax.axis('equal')
        norm = mpl.colors.Normalize(vmin=offset, vmax=offset+nW)
        cbar = fig.colorbar(mpl.cm.ScalarMappable(
            cmap=seqColormap, norm=norm), ax=ax)
        cbar.set_label('Time step')
        if ax is None:
            return fig, ax

    @staticmethod
    def traj_withCluster(x, y, labels, align='e', ax=None, seqColormap='viridis', clusterColormap='tab20'):
        n = len(x)
        nW = len(labels)
        if align == 's':
            offset = 0
        elif align == 'e':
            offset = n - nW
        elif align == 'c':
            offset = (n - nW)//2
        else:
            raise ValueError('align must be e, s, or c')

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        # plot line
        ax.plot(x, y, 'k', alpha=0.3)

        # plot starting point
        ax.plot(x[0], y[0], 'dr')

        # plot labels
        cmap = mpl.cm.get_cmap(clusterColormap)
        nCluster = len(set(labels))+1
        edgecolors = cmap(labels)
        sc = ax.scatter(x[offset:offset+nW], y[offset:offset+nW],
                        edgecolors=edgecolors,
                        s=200,
                        linewidths=2,
                        facecolors='none',
                        alpha=0.8)

        # plot sample points with color
        cmap = mpl.cm.get_cmap(seqColormap)
        colors = cmap(range(n))
        ax.scatter(x, y, c=colors, s=20, alpha=1)
        ax.axis('equal')
        if ax is None:
            return fig, ax

    @staticmethod
    def fig2img(fig):
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = PIL.Image.fromarray(img)
        return img

    # ---------------------------------------------------------------------------- #
    #                           Plot traj_and_Reconstruc                           #
    # ---------------------------------------------------------------------------- #
    @staticmethod
    def traj_and_Reconstruc_from_batch(x, y, x_full=None, fig=None, nSegment=24, nCol=5, cmap='viridis'):
        ''' First order function for plotting trajectory and reconstructed trajectory
        '''
        nBatch = x.shape[0]
        wSize = x.shape[1]
        plot_offset = 0

        # compute starting points of segments
        start_idx = np.linspace(0, nBatch, nSegment + 1).astype(int)[:-1]
        nRow = np.ceil((nSegment+1) / nCol).astype(int)

        # setup colormap
        if x_full is None:
            t_len = 301
        else:
            t_len = x_full.shape[0]
        colors = np.linspace(0, 1, t_len)
        cmap = mpl.cm.get_cmap(cmap)

        # setup figure
        if fig is None:
            fig = plt.figure(figsize=(4*nRow, 4*nCol))
        ax = fig.subplots(nRow, nCol)
        if ax.ndim == 1:
            ax = ax.reshape(1, -1)

        # plot full trajectory
        if x_full is not None:
            ax[0, 0].plot(x_full[:, 0], x_full[:, 1], '-k', alpha=0.2)
            ax[0, 0].scatter(x_full[:, 0], x_full[:, 1], c=colors, cmap=cmap)
            ax[0, 0].plot(0, 0, 'Dr', label='start', markersize=8)
            ax[0, 0].axis('equal')
            plot_offset = 1

        for i, si in enumerate(start_idx):
            iRow, iCol = np.unravel_index(i+plot_offset, (nRow, nCol))

            # plot ground Truth
            ax[iRow, iCol].scatter(x[si, :, 0], x[si, :, 1], c=cmap(
                colors[si:si+wSize]), marker='o')
            ax[iRow, iCol].plot(x[si, :, 0], x[si, :, 1], 'k', alpha=0.5)
            ax[iRow, iCol].axis('equal')

            # plot reconstructed
            ax[iRow, iCol].plot(y[si, 0, 0], y[si, 0, 1],
                                'ro', mfc='none', markersize=10)
            ax[iRow, iCol].plot(y[si, :, 0], y[si, :, 1],
                                color='red', alpha=0.5)
            ax[iRow, iCol].plot(y[si, :, 0], y[si, :, 1],
                                '.', color='red', alpha=0.5)
            ax[iRow, iCol].axis('equal')
            ax[iRow, iCol].set_title(f'{si/60:.1f}s~{(si+wSize)/60:.1f}s')

        return fig, ax

    @staticmethod
    def traj_and_Reconstruc_from_trial(df, trialno, model, wSize=30, **kwargs):
        '''
        Second order function for plotting trajectory and reconstructed trajectory
        Model is run at this level to get the reconstructed trajectory
        '''
        # extract data
        df = df.query(f'trialno == {trialno}')
        x = DataProcessing.rollingWindow_from_df(df, wSize, 1)

        # run reconstruction
        model.eval()
        x_ = torch.from_numpy(x).double()
        y = model(x_).detach().cpu().numpy()

        # cumsum
        x_cum = x.cumsum(axis=1)
        y_cum = y.cumsum(axis=1)
        x_full = df[['x-shift', 'y-shift']].values
        x_full = x_full.cumsum(axis=0)

        return Plot.traj_and_Reconstruc_from_batch(x_cum, y_cum, x_full=x_full, **kwargs)

    @staticmethod
    def traj_and_Reconstruc_quick_check(subj, task, trialno, path='TrajNet_train', model_type='val', **kwargs):
        '''Third order function for plotting trajectory and reconstructed trajectory
        '''
        # load data
        df = LoadData.mouseMovement(subj, task)

        # load model
        model = Model.load(subj=subj, task=task, path=path,
                           model_type=model_type)
        return Plot.traj_and_Reconstruc_from_trial(df, trialno=trialno, model=model, **kwargs)

    @staticmethod
    def traj_and_Reconstruc(x, y, ax, legend=True):
        """ plot trajectory and reconstructed trajectory simple version 
        Args:
            x: Ground true trajectory
            y: Reconstructed
            ax: matplotlib axis
        """

        x = np.vstack([np.zeros((1, 2)), x])
        y = np.vstack([np.zeros((1, 2)), y])
        ax.plot(x[:, 0], x[:, 1], '-')
        ax.plot(y[:, 0], y[:, 1], '-')
        ax.plot(0, 0, 'or')
        ax.axis('equal')
        if legend:
            ax.legend(['Ground true trajectory', 'Reconstructed trajectory', 'orig'],
                      bbox_to_anchor=(1.05, 1), loc=2)


class Model:

    @staticmethod
    def load(subj='K-Reg-S-18', task='one_dot', model_type='val', path='TrajNet_train_onUse'):
        ''' Load model from checkpoint
        '''
        import TrajNet_train
        model = TrajNet_train.PL_model()
        path_cp = path_data / path / f'{subj}_{task}_{model_type}.ckpt'
        model = model.load_from_checkpoint(path_cp).double().eval()
        return model
    
    
    @staticmethod
    def quick_forward(subj, x):
        ''' passing x to subj's model
        x can be a list of numpy array or a numpy array
        '''
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Model.load(subj).to(device)
        isList = type(x) is list
        if not isList:
            x = [x]            
        h = []
        y = []
        for x_ in x:
            x_ = torch.from_numpy(x_).double().to(device)
            y_ = model.forward(x_)
            h_ = model.model.x_hidden
            y.append(y_.detach().cpu().numpy())
            h.append(h_.detach().cpu().numpy())   
        if not isList:
            y = y[0] 
            h = h[0]
        return h, y    


class Analysis:
    @staticmethod
    def fit_function(x, y, fun=expon, plot=False):
        from scipy.optimize import curve_fit
        para = curve_fit(fun, x, y, 0.5)
        if plot:
            plt.plot(x, fun(x, *para[0]))
            plt.plot(x, y)
        return para

    @staticmethod
    def pca(x, n_components=None, normalise=True, plot_explained_variance=False):
        from sklearn.decomposition import PCA
        if normalise:
            from sklearn.preprocessing import scale
            x = scale(x, axis=0)
        pca = PCA(n_components=n_components)
        pca.fit(x)
        if plot_explained_variance:
            n = len(pca.explained_variance_ratio_)
            plt.bar(range(n), pca.explained_variance_ratio_.cumsum())
            plt.xlabel('Number of components')
            plt.ylabel('Cumulative explained variance')
            plt.show()
        return pca

    @staticmethod
    def dim_measure(x):
        # perform PCA and fit exponential distribution to explained_variance_ratio_
        pca = Analysis.pca(x)
        y = pca.explained_variance_ratio_
        x = np.arange(len(y))
        return Analysis.fit_function(x, y)[0][0]
    
    @staticmethod
    def auc_oneVsOthers(x):
        ''' compute AUC for one vs others
        x: sample x class 
        '''
        from sklearn import metrics 
        auc = []
        for i in range(x.shape[1]):
            y_true = np.zeros(x.shape)
            y_true[:, i] = 1
            fpr, tpr, thresholds = metrics.roc_curve(y_true.flatten(), -x.flatten())
            auc.append(metrics.auc(fpr, tpr))
        return np.hstack(auc)    
    
    @staticmethod
    def argmin_ratio(x):
        ''' the ratio of the class with minimal value at each sample point
        x: sample x class 
        '''
        iMin = x.argmin(axis=1)
        unique, counts = np.unique(iMin, return_counts=True)
        b = np.zeros(3)
        for i, j in zip(unique, counts):
            b[i] = j
        return b / x.shape[0]
    
    @staticmethod
    def class_in_topN(dist_timeSeries):
        n, nc = dist_timeSeries.shape
        labels = np.ones_like(dist_timeSeries) * np.arange(nc)
        topN = np.argsort(dist_timeSeries.flatten())[0:n]
        topN = labels.flatten()[topN]
        ratio = [np.sum(topN == i) / n  for i in range(nc)]
        return ratio

    @staticmethod
    def rsa(X, dist_measure='euclidean'):
        '''
        X: list of numpy arrays. 
        X[i] i is subjects.
        X[i] is a 2D array samples x fetures
        First, we calculate distance matrix between each pair of samples for each subject
        Then, we compute the similarity of the distance matrix between each subjects
        Finally, return the similarity matrix
        '''
        from sklearn.metrics import pairwise_distances
        dist_mat = [pairwise_distances(x, metric=dist_measure).flatten() for x in X]
        dist_mat = np.vstack(dist_mat)
        return np.corrcoef(dist_mat)    

class GroupOperation:

    @staticmethod
    def map(fun, subjs, *args, **kwargs):
        data = []
        with alive_bar(len(subjs), force_tty=True, title='Group loop') as bar:
            for i, subj in enumerate(subjs):
                data.append(fun(subj, *args, **kwargs))
                bar()
        return data


    @staticmethod
    def map_trial(fun, trials):
        ''' run trial loop for funtion with iTrial as input
        trials: list of trial numbers
        '''
        data = []
        # with alive_bar(len(trials), force_tty=True, title='Trial loop') as bar:
        for trial in trials:
            data.append(fun(trial))
            # bar()
        return data

class test:
    @staticmethod
    def quick_forward(subj, x):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Model.load(subj).to(device)
        if type(x) is not list:
            x = [x]
            
        h = []
        y = []
        for x_ in x:
            x_ = torch.from_numpy(x_).double().to(device)
            y_ = model.forward(x_)
            h_ = model.model.x_hidden
            y.append(y_.detach().cpu().numpy())
            h.append(h_.detach().cpu().numpy())    
        if type(x) is not list:
            y = y[0] 
            h = h[0]
        return h, y


class Save:
    @staticmethod
    def savepath(folder, filename):
        pathname = path_data / folder
        pathname.mkdir(parents=True, exist_ok=True)
        return str(pathname / filename)

# manuscript class
class ms:
    path = path_project / 'ms'
    path_fig = path / 'fig'
    def __init__(self):
        pass