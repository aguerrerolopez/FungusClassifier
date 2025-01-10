import numpy as np
import pandas as pd
from scipy.stats import binned_statistic
import os
from scipy.signal import savgol_filter
from scipy import sparse
from scipy.linalg import norm
import pandas as pd
import numpy as np
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Code copied from maldi-nn, thanks to the authors. Adapted by Alejandro Guerrero-López.


class SpectrumObject:
    """Base Spectrum Object class

    Can be instantiated directly with 1-D np.arrays for mz and intensity.
    Alternatively, can be read from csv files or from bruker output data.
    Reading from Bruker data is based on the code in https://github.com/sgibb/readBrukerFlexData

    Parameters
    ----------
    mz : 1-D np.array, optional
        mz values, by default None
    intensity : 1-D np.array, optional
        intensity values, by default None
    """

    def __init__(self, mz=None, intensity=None):
        self.mz = mz
        self.intensity = intensity
        if self.intensity is not None:
            if np.issubdtype(self.intensity.dtype, np.unsignedinteger):
                self.intensity = self.intensity.astype(int)
        if self.mz is not None:
            if np.issubdtype(self.mz.dtype, np.unsignedinteger):
                self.mz = self.mz.astype(int)

    def __getitem__(self, index):
        return SpectrumObject(mz=self.mz[index], intensity=self.intensity[index])

    def __len__(self):
        if self.mz is not None:
            return self.mz.shape[0]
        else:
            return 0

    def plot(self, as_peaks=False, **kwargs):
        """Plot a spectrum via matplotlib

        Parameters
        ----------
        as_peaks : bool, optional
            draw points in the spectrum as individualpeaks, instead of connecting the points in the spectrum, by default False
        """
        if as_peaks:
            mz_plot = np.stack([self.mz - 1, self.mz, self.mz + 1]).T.reshape(-1)
            int_plot = np.stack(
                [
                    np.zeros_like(self.intensity),
                    self.intensity,
                    np.zeros_like(self.intensity),
                ]
            ).T.reshape(-1)
        else:
            mz_plot, int_plot = self.mz, self.intensity
        plt.plot(mz_plot, int_plot, **kwargs)

    def __repr__(self):
        string_ = np.array2string(
            np.stack([self.mz, self.intensity]), precision=5, threshold=10, edgeitems=2
        )
        mz_string, int_string = string_.split("\n")
        mz_string = mz_string[1:]
        int_string = int_string[1:-1]
        return "SpectrumObject([\n\tmz  = %s,\n\tint = %s\n])" % (mz_string, int_string)

    @staticmethod
    def tof2mass(ML1, ML2, ML3, TOF):
        A = ML3
        B = np.sqrt(1e12 / ML1)
        C = ML2 - TOF

        if A == 0:
            return (C * C) / (B * B)
        else:
            return ((-B + np.sqrt((B * B) - (4 * A * C))) / (2 * A)) ** 2

    @classmethod
    def from_bruker(cls, acqu_file, fid_file):
        """Read a spectrum from Bruker's format

        Parameters
        ----------
        acqu_file : str
            "acqu" file bruker folder
        fid_file : str
            "fid" file in bruker folder

        Returns
        -------
        SpectrumObject
        """
        with open(acqu_file, "rb") as f:
            lines = [line.decode("utf-8", errors="replace").rstrip() for line in f]
        for l in lines:
            if l.startswith("##$TD"):
                TD = int(l.split("= ")[1])
            if l.startswith("##$DELAY"):
                DELAY = int(l.split("= ")[1])
            if l.startswith("##$DW"):
                DW = float(l.split("= ")[1])
            if l.startswith("##$ML1"):
                ML1 = float(l.split("= ")[1])
            if l.startswith("##$ML2"):
                ML2 = float(l.split("= ")[1])
            if l.startswith("##$ML3"):
                ML3 = float(l.split("= ")[1])
            if l.startswith("##$BYTORDA"):
                BYTORDA = int(l.split("= ")[1])
            if l.startswith("##$NTBCal"):
                NTBCal = l.split("= ")[1]

        intensity = np.fromfile(fid_file, dtype={0: "<i", 1: ">i"}[BYTORDA])

        if len(intensity) < TD:
            TD = len(intensity)
        TOF = DELAY + np.arange(TD) * DW

        mass = cls.tof2mass(ML1, ML2, ML3, TOF)

        intensity[intensity < 0] = 0

        return cls(mz=mass, intensity=intensity)

    @classmethod
    def from_tsv(cls, file, sep=" "):
        """Read a spectrum from txt

        Parameters
        ----------
        file : str
            path to csv file
        sep : str, optional
            separator in the file, by default " "

        Returns
        -------
        SpectrumObject
        """
        s = pd.read_table(
            file, sep=sep, index_col=None, comment="#", header=None
        ).values
        mz = s[:, 0]
        intensity = s[:, 1]
        return cls(mz=mz, intensity=intensity)

    
class Binner:

    """Pre-processing function for binning spectra in equal-width bins.

    Parameters
    ----------
    start : int, optional
        start of the binning range, by default 2000
    stop : int, optional
        end of the binning range, by default 20000
    step : int, optional
        width of every bin, by default 3
    aggregation : str, optional
        how to aggregate intensity values in each bin.
        Is passed to the statistic argument of https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binned_statistic.html
        Can take any argument that the statistic argument also takes, by default "sum"
    """

    def __init__(self, start=2000, stop=20000, step=3, aggregation="sum"):
        self.bins = np.arange(start, stop + 1e-8, step)
        self.mz_bins = self.bins[:-1] + step / 2
        self.agg = aggregation

    def __call__(self, SpectrumObj):
        if self.agg == "sum":
            bins, _ = np.histogram(
                SpectrumObj.mz, self.bins, weights=SpectrumObj.intensity
            )
        else:
            bins = binned_statistic(
                SpectrumObj.mz,
                SpectrumObj.intensity,
                bins=self.bins,
                statistic=self.agg,
            ).statistic
            bins = np.nan_to_num(bins)

        s = SpectrumObject(intensity=bins, mz=self.mz_bins)
        return s
    
class MaldiDataset:
    def __init__(self, root_dir, n_step=3):
        self.root_dir = root_dir
        self.n_step = n_step
        self.data = []

    def parse_dataset(self):
        for folder in os.listdir(self.root_dir):
            folder_path = os.path.join(self.root_dir, folder)
            if os.path.isdir(folder_path):
                # Parse folder name for genus, species, and hospital code
                genus_species, hospital_code = self._parse_folder_name(folder)
                genus = genus_species.split()[0]
                genus_species_label = genus_species
                unique_id_label = hospital_code

                # Iterate over each replicate folder
                for replicate_folder in os.listdir(folder_path):
                    replicate_folder_path = os.path.join(folder_path, replicate_folder)
                    if os.path.isdir(replicate_folder_path):
                        # Iterate over each lecture folder
                        for lecture_folder in os.listdir(replicate_folder_path):
                            lecture_folder_path = os.path.join(replicate_folder_path, lecture_folder)
                            if os.path.isdir(lecture_folder_path):
                                # Search for "acqu" and "fid" files
                                acqu_file, fid_file = self._find_acqu_fid_files(lecture_folder_path)
                                if acqu_file and fid_file:

                                    # Read the maldi-tof spectra using from_bruker
                                    spectrum = SpectrumObject.from_bruker(acqu_file, fid_file)

                                    # Apply Gaussian smoothing to raw spectrum
                                    smoothed_intensity = gaussian_filter1d(spectrum.intensity, sigma=1)
                                    spectrum = SpectrumObject(mz=spectrum.mz, intensity=smoothed_intensity)
                                    
                                    # Binarize the spectrum using Binner
                                    binner = SequentialPreprocessor(
                                                VarStabilizer(method="sqrt"),
                                                Smoother(halfwindow=10),
                                                BaselineCorrecter(method="SNIP", snip_n_iter=20),
                                                Trimmer(),
                                                Binner(step=self.n_step),
                                                Normalizer(sum=1),
                                            )
                                    binned_spectrum = binner(spectrum)
                                    # Append data point to the dataset
                                    # if the spectrum is nan due to the preprocessing, skip it
                                    if np.isnan(binned_spectrum.intensity).any():
                                        print("Skipping nan spectrum")
                                        continue
                                    self.data.append({
                                        'spectrum': binned_spectrum.intensity,
                                        'm/z': binned_spectrum.mz,
                                        'unique_id_label': unique_id_label,
                                        'genus_label': genus,
                                        'genus_species_label': genus_species_label
                                    })
        print("Parsed dataset samples:")
        for idx, sample in enumerate(self.data):
            print(f"Sample {idx}: {sample}")

        X = np.array([entry['spectrum'] for entry in self.data])
        y = np.array([entry['genus_species_label'] for entry in self.data])

        # Handle imbalance using the chosen method
        X_balanced, y_balanced = handle_imbalance(X, y, method="VAE")

       # Update self.data with balanced data
        self.data = [{
            'spectrum': spectrum,
            'genus_species_label': label,
            'unique_id_label': f"synthetic_{i}"  # unique ID in Synthetic data
        } for i, (spectrum, label) in enumerate(zip(X_balanced, y_balanced))]

        print(f"Balanced dataset samples: {len(self.data)}") 

    def _parse_folder_name(self, folder_name):
        # Split folder name into genus, species, and hospital code
        parts = folder_name.split()
        genus_species = " ".join(parts[:2])
        hospital_code = " ".join(parts[2:])
        return genus_species, hospital_code

    def _find_acqu_fid_files(self, directory):
        acqu_file = None
        fid_file = None
        for root, _, files in os.walk(directory):
            for file in files:
                if file == 'acqu':
                    acqu_file = os.path.join(root, file)
                elif file == 'fid':
                    fid_file = os.path.join(root, file)
                if acqu_file and fid_file:
                    return acqu_file, fid_file
        return acqu_file, fid_file

    def get_data(self):
        return self.data



class Normalizer:
    """Pre-processing function for normalizing the intensity of a spectrum.
    Commonly referred to as total ion current (TIC) calibration.

    Parameters
    ----------
    sum : int, optional
        Make the total intensity of the spectrum equal to this amount, by default 1
    """

    def __init__(self, sum=1):
        self.sum = sum

    def __call__(self, SpectrumObj):
        s = SpectrumObject()

        s = SpectrumObject(
            intensity=SpectrumObj.intensity / SpectrumObj.intensity.sum() * self.sum,
            mz=SpectrumObj.mz,
        )
        return s


class Trimmer:
    """Pre-processing function for trimming ends of a spectrum.
    This can be used to remove inaccurate measurements.

    Parameters
    ----------
    min : int, optional
        remove all measurements with mz's lower than this value, by default 2000
    max : int, optional
        remove all measurements with mz's higher than this value, by default 20000
    """

    def __init__(self, min=2000, max=20000):
        self.range = [min, max]

    def __call__(self, SpectrumObj):
        indices = (self.range[0] < SpectrumObj.mz) & (SpectrumObj.mz < self.range[1])

        s = SpectrumObject(
            intensity=SpectrumObj.intensity[indices], mz=SpectrumObj.mz[indices]
        )
        return s


class VarStabilizer:
    """Pre-processing function for manipulating intensities.
    Commonly performed to stabilize their variance.

    Parameters
    ----------
    method : str, optional
        function to apply to intensities.
        can be either "sqrt", "log", "log2" or "log10", by default "sqrt"
    """

    def __init__(self, method="sqrt"):
        methods = {"sqrt": np.sqrt, "log": np.log, "log2": np.log2, "log10": np.log10}
        self.fun = methods[method]

    def __call__(self, SpectrumObj):
        s = SpectrumObject(intensity=self.fun(SpectrumObj.intensity), mz=SpectrumObj.mz)
        return s


class BaselineCorrecter:
    """Pre-processing function for baseline correction (also referred to as background removal).

    Support SNIP, ALS and ArPLS.
    Some of the code is based on https://stackoverflow.com/questions/29156532/python-baseline-correction-library.

    Parameters
    ----------
    method : str, optional
        Which method to use
        either "SNIP", "ArPLS" or "ALS", by default None
    als_lam : float, optional
        lambda value for ALS and ArPLS, by default 1e8
    als_p : float, optional
        p value for ALS and ArPLS, by default 0.01
    als_max_iter : int, optional
        max iterations for ALS and ArPLS, by default 10
    als_tol : float, optional
        stopping tolerance for ALS and ArPLS, by default 1e-6
    snip_n_iter : int, optional
        iterations of SNIP, by default 10
    """

    def __init__(
        self,
        method=None,
        als_lam=1e8,
        als_p=0.01,
        als_max_iter=10,
        als_tol=1e-6,
        snip_n_iter=10,
    ):
        self.method = method
        self.lam = als_lam
        self.p = als_p
        self.max_iter = als_max_iter
        self.tol = als_tol
        self.n_iter = snip_n_iter

    def __call__(self, SpectrumObj):
        if "LS" in self.method:
            baseline = self.als(
                SpectrumObj.intensity,
                method=self.method,
                lam=self.lam,
                p=self.p,
                max_iter=self.max_iter,
                tol=self.tol,
            )
        elif self.method == "SNIP":
            baseline = self.snip(SpectrumObj.intensity, self.n_iter)

        s = SpectrumObject(
            intensity=SpectrumObj.intensity - baseline, mz=SpectrumObj.mz
        )
        return s

    def als(self, y, method="ArPLS", lam=1e8, p=0.01, max_iter=10, tol=1e-6):
        L = len(y)
        D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
        D = lam * D.dot(
            D.transpose()
        )  # Precompute this term since it does not depend on `w`

        w = np.ones(L)
        W = sparse.spdiags(w, 0, L, L)

        crit = 1
        count = 0
        while crit > tol:
            z = sparse.linalg.spsolve(W + D, w * y)

            if method == "AsLS":
                w_new = p * (y > z) + (1 - p) * (y < z)
            elif method == "ArPLS":
                d = y - z
                dn = d[d < 0]
                m = np.mean(dn)
                s = np.std(dn)
                w_new = 1 / (1 + np.exp(np.minimum(2 * (d - (2 * s - m)) / s, 70)))

            crit = norm(w_new - w) / norm(w)
            w = w_new
            W.setdiag(w)
            count += 1
            if count > max_iter:
                break
        return z

    def snip(self, y, n_iter):
        y_prepr = np.log(np.log(np.sqrt(y + 1) + 1) + 1)
        for i in range(1, n_iter + 1):
            rolled = np.pad(y_prepr, (i, i), mode="edge")
            new = np.minimum(
                y_prepr, (np.roll(rolled, i) + np.roll(rolled, -i))[i:-i] / 2
            )
            y_prepr = new
        return (np.exp(np.exp(y_prepr) - 1) - 1) ** 2 - 1


class Smoother:
    """Pre-processing function for smoothing. Uses Savitzky-Golay filter.

    Parameters
    ----------
    halfwindow : int, optional
        halfwindow of savgol_filter, by default 10
    polyorder : int, optional
        polyorder of savgol_filter, by default 3
    """

    def __init__(self, halfwindow=10, polyorder=3):
        self.window = halfwindow * 2 + 1
        self.poly = polyorder

    def __call__(self, SpectrumObj):
        s = SpectrumObject(
            intensity=np.maximum(
                savgol_filter(SpectrumObj.intensity, self.window, self.poly), 0
            ),
            mz=SpectrumObj.mz,
        )
        return s



class LocalMaximaPeakDetector:
    """
    Detects peaks a la MaldiQuant

    Parameters
    ----------
    SNR : int, optional
        Signal to noise radio. This function computes a SNR value as the median absolute deviation from the median intensity (MAD).
        Only peaks with intensities a multiple of this SNR are considered. By default 2.
    halfwindowsize: int, optional
        half window size, an intensity can only be a peak if it is the highest value in a window. By default 20, for a total window size of 41.
    """

    def __init__(
        self,
        SNR=2,
        halfwindowsize=20,
    ):
        self.hw = halfwindowsize
        self.SNR = SNR

    def __call__(self, SpectrumObj):
        SNR = (
            np.median(np.abs(SpectrumObj.intensity - np.median(SpectrumObj.intensity)))
            * self.SNR
        )

        local_maxima = np.argmax(
            np.lib.stride_tricks.sliding_window_view(
                SpectrumObj.intensity, (int(self.hw * 2 + 1),)
            ),
            -1,
        ) == int(self.hw)
        s_int_local = SpectrumObj.intensity[self.hw : -self.hw][local_maxima]
        s_mz_local = SpectrumObj.mz[self.hw : -self.hw][local_maxima]
        return SpectrumObject(
            intensity=s_int_local[s_int_local > SNR], mz=s_mz_local[s_int_local > SNR]
        )


class PeakFilter:
    """Pre-processing function for filtering peaks.

    Filters in two ways: absolute number of peaks and height.

    Parameters
    ----------
    max_number : int, optional
        Maximum number of peaks to keep. Prioritizes peaks to keep by height.
        by default None, for no filtering
    min_intensity : float, optional
        Min intensity of peaks to keep, by default None, for no filtering
    """

    def __init__(self, max_number=None, min_intensity=None):
        self.max_number = max_number
        self.min_intensity = min_intensity

    def __call__(self, SpectrumObj):
        s = SpectrumObject(intensity=SpectrumObj.intensity, mz=SpectrumObj.mz)

        if self.max_number is not None:
            indices = np.argsort(-s.intensity, kind="stable")
            take = np.sort(indices[: self.max_number])

            s.mz = s.mz[take]
            s.intensity = s.intensity[take]

        if self.min_intensity is not None:
            take = s.intensity >= self.min_intensity

            s.mz = s.mz[take]
            s.intensity = s.intensity[take]

        return s


class RandomPeakShifter:
    """Pre-processing function for adding random (gaussian) noise to the mz values of peaks.

    Parameters
    ----------
    std : float, optional
        stdev of the random noise to add, by default 1
    """

    def __init__(self, std=1.0):
        self.std = std

    def __call__(self, SpectrumObj):
        s = SpectrumObject(
            intensity=SpectrumObj.intensity,
            mz=SpectrumObj.mz
            + np.random.normal(scale=self.std, size=SpectrumObj.mz.shape),
        )
        return s


class UniformPeakShifter:
    """Pre-processing function for adding uniform noise to the mz values of peaks.

    Parameters
    ----------
    range : float, optional
        let each peak shift by maximum this value, by default 1.5
    """

    def __init__(self, range=1.5):
        self.range = range

    def __call__(self, SpectrumObj):
        s = SpectrumObject(
            intensity=SpectrumObj.intensity,
            mz=SpectrumObj.mz
            + np.random.uniform(
                low=-self.range, high=self.range, size=SpectrumObj.mz.shape
            ),
        )
        return s


class Binarizer:
    """Pre-processing function for binarizing intensity values of peaks.

    Parameters
    ----------
    threshold : float
        Threshold for the intensities to become 1 or 0.
    """

    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, SpectrumObj):
        s = SpectrumObject(
            intensity=(SpectrumObj.intensity > self.threshold).astype(
                SpectrumObj.intensity.dtype
            ),
            mz=SpectrumObj.mz,
        )
        return s

#VAE
def train_vae(X, latent_dim=10, epochs=10, batch_size=16):
    if X is None or len(X) == 0:
        raise ValueError("Input data X is empty or None.")
    
    if len(X.shape) != 2:
        raise ValueError(f"Input data X must be 2D, but got shape {X.shape}.")

    input_dim = X.shape[1]

    # Convert X to PyTorch tensor
    X_tensor = torch.tensor(X, dtype=torch.float32)
    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize the VAE model
    vae = VAE(input_dim, latent_dim)
    optimizer = optim.Adam(vae.parameters(), lr=0.001)

    # Training loop
    for epoch in range(epochs):
        for batch in dataloader:
            x_batch = batch[0]
            reconstructed, z_mean, z_log_var = vae(x_batch)

            # Reconstruction loss (MSE)
            reconstruction_loss = nn.functional.mse_loss(reconstructed, x_batch, reduction='sum')

            # KL divergence
            kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())

            # Total loss
            loss = reconstruction_loss + kl_loss

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    return vae


def generate_synthetic_data_with_vae(X, y, num_samples_per_class=10, latent_dim=10):
    unique_classes = np.unique(y)
    synthetic_X = []
    synthetic_y = []

    for cls in unique_classes:
        # Extract data for the current class
        class_data = X[y == cls]

        if len(class_data) < 2:
            # Skip classes with insufficient data
            print(f"Skipping class {cls} due to insufficient samples.")
            continue

        # Ensure class_data is 2D
        if len(class_data.shape) == 1:
            class_data = class_data.reshape(-1, 1)

        # Train a VAE for the current class
        vae = train_vae(class_data, latent_dim=latent_dim)

        # Generate synthetic data
        z_samples = torch.randn(num_samples_per_class, latent_dim)
        with torch.no_grad():
            generated_data = vae.decode(z_samples)
        # Ensure generated_data is a numpy array
        if isinstance(generated_data, list):
            generated_data = np.array(generated_data)

        # Debugging output
        print(f"Generated data shape: {generated_data.shape}")
        # Ensure generated_data is converted to numpy array
        if isinstance(generated_data, torch.Tensor):
            generated_data = generated_data.cpu().numpy()  # Convert to numpy array if torch tensor
        elif isinstance(generated_data, list):
            generated_data = np.array(generated_data)  # Convert list to numpy array

        # Ensure generated data has the same number of features as the original class data
        if generated_data.shape[1] != X.shape[1]:
            generated_data = np.tile(generated_data, (1, X.shape[1] // generated_data.shape[1] + 1))[:, :X.shape[1]]

        synthetic_X.append(generated_data)
        synthetic_y.extend([cls] * num_samples_per_class)

    # Combine all synthetic data
    if len(synthetic_X) == 0:
        print("No synthetic data generated. Returning original data.")
        return X, y  # Return original data if no synthetic data generated.

    synthetic_X = np.vstack(synthetic_X)
    synthetic_y = np.array(synthetic_y)

    return synthetic_X, synthetic_y


def handle_imbalance(X, y, method="VAE", latent_dim=10, num_samples_per_class=10):
    if method == "VAE":
        synthetic_X, synthetic_y = generate_synthetic_data_with_vae(
            X, y, latent_dim=latent_dim, num_samples_per_class=num_samples_per_class
        )
        return np.vstack((X, synthetic_X)), np.hstack((y, synthetic_y))
    else:
        raise ValueError(f"Unsupported method: {method}. Currently supports only 'VAE'.")



class SequentialPreprocessor:
    """Chain multiple preprocessors so that a pre-processing pipeline can be called with one line.

    Example:
    ```python
    preprocessor = SequentialPreprocessor(
        VarStabilizer(),
        Smoother(),
        BaselineCorrecter(method="SNIP"),
        Normalizer(),
        Binner()
    )
    preprocessed_spectrum = preprocessor(spectrum)
    ```
    """

    def __init__(self, *args):
        self.preprocessors = args

    def __call__(self, SpectrumObj):
        for step in self.preprocessors:
            SpectrumObj = step(SpectrumObj)
        return SpectrumObj
