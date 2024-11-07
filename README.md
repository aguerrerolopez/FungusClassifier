# FungusClassifier


# Data Reader Functions

The `data_reader` module in this project provides various functions for reading and preprocessing spectral data. Below is an overview of the available functions:

### Reading Data

- **`from_bruker(acqu_file, fid_file)`**: Reads a spectrum from Bruker files, taking the "acqu" and "fid" files as inputs. This function uses metadata to properly calculate the mass/charge (m/z) values and extract intensity data, allowing for a comprehensive SpectrumObject.

- **`from_tsv(file, sep=" ")`**: Reads a spectrum from a tab-separated value file, extracting the m/z and intensity values from the first two columns.

### Preprocessing Functions

- **`Binner(start=2000, stop=20000, step=3, aggregation="sum")`**: Bins spectra into equal-width intervals, aggregating intensities using the specified method.

- **`Normalizer(sum=1)`**: Normalizes the intensity values to ensure the total intensity is equal to the specified sum (default is 1).

- **`Trimmer(min=2000, max=20000)`**: Trims m/z values outside the specified range, removing inaccurate measurements.

- **`VarStabilizer(method="sqrt")`**: Applies a transformation to stabilize variance, using methods like square root, log, log2, or log10.

- **`BaselineCorrecter(method="SNIP", ...)`**: Corrects the baseline using SNIP, ALS, or ArPLS methods, removing background noise from spectra.

- **`Smoother(halfwindow=10, polyorder=3)`**: Smooths the spectrum using a Savitzky-Golay filter to reduce noise.

- **`LocalMaximaPeakDetector(SNR=2, halfwindowsize=20)`**: Detects peaks by finding local maxima and using a signal-to-noise ratio threshold.

- **`PeakFilter(max_number=None, min_intensity=None)`**: Filters peaks by height or limits the number of peaks based on specified criteria.

- **`RandomPeakShifter(std=1.0)`**: Adds random Gaussian noise to the m/z values of peaks to simulate variability.

- **`UniformPeakShifter(range=1.5)`**: Adds uniform noise to the m/z values of peaks within the specified range.

- **`Binarizer(threshold)`**: Converts intensity values to binary (0 or 1) based on a specified threshold.

- **`SequentialPreprocessor(*args)`**: Chains multiple preprocessing steps into one callable pipeline for ease of use. For example, this allows applying variance stabilization, smoothing, baseline correction, normalization, binning, etc., in sequence.

### Typical Preprocessing Order Example

A typical order of preprocessing steps using `SequentialPreprocessor` might look like this:

```python
SequentialPreprocessor(
    VarStabilizer(method="sqrt"),
    Smoother(halfwindow=10),
    BaselineCorrecter(method="SNIP", snip_n_iter=20),
    Trimmer(),
    Binner(step=self.n_step),
    Normalizer(sum=1),
)
