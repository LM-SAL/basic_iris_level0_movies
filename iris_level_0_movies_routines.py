import logging
import numbers
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.visualization import (
    AsinhStretch,
)
from astropy.visualization.mpl_normalize import ImageNormalize
from matplotlib.animation import FFMpegWriter, FuncAnimation
from scipy import ndimage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_regions(array, min_size=1):
    """Extracts separate regions of non-NaN values from array.

    Parameters
    ----------
    array : np.ndarray
        Input array with NaN and non-NaN regions.
    min_size : int, optional
        Minimum size of region to extract, by default 1.

    Returns
    -------
    list
        List of arrays containing each region

    """
    mask = ~np.isnan(array)
    labeled_array, num_features = ndimage.label(mask)
    regions = []
    for label in range(1, num_features + 1):
        region_mask = labeled_array == label
        if np.sum(region_mask) >= min_size:
            rows, cols = np.where(region_mask)
            region = array[min(rows) : max(rows) + 1, min(cols) : max(cols) + 1]
            regions.append(region)
    return regions


def combine_regions(regions, spacing=1):
    """Combines multiple regions into a single array with spacing between them.

    Parameters
    ----------
    regions : list
        List of arrays containing each region.
    spacing : int, optional
        Number of NaN columns to add between regions, by default 1.

    Returns
    -------
    np.ndarray
        Combined array with all regions.

    """
    if not regions:
        return np.array([])
    max_height = max(r.shape[0] for r in regions)
    padded_regions = []
    for region in regions:
        if region.shape[0] < max_height:
            pad_top = (max_height - region.shape[0]) // 2
            pad_bottom = max_height - region.shape[0] - pad_top
            region = np.pad(
                region,
                ((pad_top, pad_bottom), (0, 0)),
                constant_values=np.nan,
            )
        padded_regions.append(region)
    if spacing > 0:
        spacer = np.full((max_height, spacing), np.nan)
        combined = padded_regions[0]
        for region in padded_regions[1:]:
            combined = np.concatenate([combined, spacer, region], axis=1)
    else:
        combined = np.concatenate(padded_regions, axis=1)
    return combined


def image_clipping(image, cutoff=1.5e-3, gamma=1.0):
    """Computes and returns the min and max values of the input (image), clipping
    brightest and darkest pixels.

    Parameters
    ----------
    image : `numpy.ndarray`
        The input image.
    cutoff : float, optional
        The cutoff value for the histogram.
        Defaults to 1.5e-3
    gamma : float, optional
        The gamma value for the histogram.
        Defaults to 1.0

    References
    ----------
    Based on original IDL routine by P.Suetterlin (06 Jul 1993)
    Ported by V.Hansteen (15 Apr 2020)

    """
    hmin = np.nanmin(image)
    hmax = np.nanmax(image)
    if issubclass(image.dtype.type, numbers.Integral):
        nbins = np.abs(np.nanmax(image) - np.nanmin(image))
        hist = np.histogram(image, bins=nbins)
        fak = 1
    else:
        nbins = 10000
        fak = nbins / (hmax - hmin)
        hist = np.histogram((image - hmin) * fak, range=(0.0, float(nbins)), bins=nbins)
    h = hist[0]
    bins = hist[1]
    nh = np.size(h)
    # Integrate the histogram so that h(i) holds the number of points
    # with equal or lower intensity.
    for i in range(1, nh - 1):
        h[i] = h[i] + h[i - 1]
    h = h / float(h[nh - 2])
    h[nh - 1] = 1
    # As cutoff is in percent and h is normalized to unity,
    # vmin/vmax are the indices of the point where the number of pixels
    # with lower/higher intensity reach the given limit. This has to be
    # converted to a real image value by dividing by the scalefactor
    # fak and adding the min value of the image
    # Note that the bottom value is taken off (addition of h[0] to cutoff),
    # there are often very many points in IRIS images that are set to zero, this
    # removes them from calculation... and seems to work.
    vmin = (
        np.nanmax(np.where(h <= (cutoff + h[0]), bins[1:] - bins[0], 0)) / fak + hmin
    ) ** gamma
    vmax = (
        np.nanmin(np.where(h >= (1.0 - cutoff), bins[1:] - bins[0], nh - 2)) / fak
        + hmin
    ) ** gamma
    return vmin, vmax


class FITSMovieMaker:
    """A class to create movies from FITS files using matplotlib."""

    def __init__(
        self,
        fits_files: list[str],
        output_path: str,
        interval: int = 500,
        fps: int = 10,
        cmap: str = "magma",
    ) -> None:
        """Initialize the FITSMovieMaker class.

        Parameters
        ----------
        fits_files : List[str]
            List of paths to FITS files.
        output_path : str
            Path for the output movie file.
        interval : int
            Interval between frames in milliseconds.
        fps : int
            Frames per second for the output movie.
        cmap : str
            Matplotlib colormap to use
        percentile : Tuple[float, float]
            Percentiles for contrast scaling.
        stretch : LinearStretch, optional
            Stretch model from astropy.

        """
        self.fits_files = fits_files
        self.output_path = Path(output_path)
        self.interval = interval
        self.fps = fps
        self.cmap = cmap

    def _read_fits(self, fits_path):
        """Read FITS file.

        We need the second HDU because the first one is for compression.

        Parameters
        ----------
        fits_path : Path
            Path to FITS file.

        Returns
        -------
        np.ndarray
            Image data.
        Header
            FITs Header.

        """
        try:
            with fits.open(fits_path) as hdul:
                regions = extract_regions(hdul[1].data)
                combined = combine_regions(regions, spacing=2)
                return combined, hdul[1].header
        except Exception:
            logger.exception(f"Error reading FITS file {fits_path}")
            raise

    def _update_table(self, header) -> None:
        """Update the table with the current frame's header values."""
        table_data = [
            [f'FSN: {header["FSN"]}'],
            [f'IMG_PATH: {header["IMG_PATH"]}'],
            [f'EXPTIME: {header["EXPTIME"]}'],
            [f'ISQOLTID: {header["ISQOLTID"]}'],
            [f'IISSLOOP: {header["IISSLOOP"]}'],
            [f'IAECEVFL: {header["IAECEVFL"]}'],
            [f'IAECFLAG: {header["IAECFLAG"]}'],
            [f'SUMSPAT: {header["SUMSPAT"]}'],
            [f'SUMSPTRL: {header["SUMSPTRL"]}'],
        ]
        for key in self.table._cells:
            cell = self.table._cells[key]
            cell.get_text().set_text(table_data[key[0]][key[1]])

    def _setup_plot(self, first_frame, first_header) -> None:
        """Set up the matplotlib figure and axis for animation."""
        modifier = "SJI" in first_header["IMG_PATH"]
        self.fig = plt.figure(figsize=(12, 12) if modifier else (16, 8))
        self.gs = plt.GridSpec(2, 1, height_ratios=[10, 1])
        self.vmin, self.vmax = image_clipping(first_frame)
        self.ax_im = plt.subplot(self.gs[0])
        self.im = self.ax_im.imshow(
            first_frame,
            norm=ImageNormalize(vmin=self.vmin, vmax=self.vmax, stretch=AsinhStretch()),
            cmap=self.cmap,
            origin="lower",
        )
        self.cb = plt.colorbar(
            self.im,
            ax=self.ax_im,
            shrink=0.5,
            fraction=0.1,
            orientation="vertical" if modifier else "horizontal",
        )
        self.ax_im.set_title(f'DATE-OBS {first_header["DATE-OBS"]}')
        self.ax_table = plt.subplot(self.gs[1])
        table_data = [
            [f'FSN: {first_header["FSN"]}'],
            [f'IMG_PATH: {first_header["IMG_PATH"]}'],
            [f'EXPTIME: {first_header["EXPTIME"]}'],
            [f'ISQOLTID: {first_header["ISQOLTID"]}'],
            [f'IISSLOOP: {first_header["IISSLOOP"]}'],
            [f'IAECEVFL: {first_header["IAECEVFL"]}'],
            [f'IAECFLAG: {first_header["IAECFLAG"]}'],
            [f'SUMSPAT: {first_header["SUMSPAT"]}'],
            [f'SUMSPTRL: {first_header["SUMSPTRL"]}'],
        ]
        self.table = self.ax_table.table(
            cellText=table_data,
            loc="center",
            cellLoc="center",
        )
        self.ax_table.axis("off")
        self.table.scale(0.2, 1)
        plt.tight_layout()

    def _update_frame(self, frame_num) -> list:
        """Update function for animation."""
        try:
            data, header = self._read_fits(self.fits_files[frame_num])
            self.im.set_data(data)
            self.im.set_clim(self.vmin, self.vmax)
            self.ax_im.set_title(f'DATE-OBS {header["DATE-OBS"]}')
            self._update_table(header)
            return [self.im]
        except Exception:
            logger.exception(f"Error updating frame {frame_num}")
            raise

    def create_movie(self) -> None:
        """Create the movie from the FITS files."""
        try:
            first_frame, first_header = self._read_fits(self.fits_files[0])
            self._setup_plot(first_frame, first_header)
            logger.info("Creating animation...")
            anim = FuncAnimation(
                self.fig,
                self._update_frame,
                frames=len(self.fits_files),
                interval=self.interval,
                blit=True,
            )
            logger.info(f"Saving animation to {self.output_path}")
            writer = FFMpegWriter(fps=self.fps)
            Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
            anim.save(self.output_path, writer=writer)
            plt.close()
            logger.info("Movie creation completed successfully!")
        except Exception:
            logger.exception("Error creating movie")
            raise


def get_date_range(start_date, end_date):
    start = datetime.strptime(start_date, "%Y/%m/%d/H%H00")
    end = datetime.strptime(end_date, "%Y/%m/%d/H%H00")
    dates = []
    while start <= end:
        dates.append(start.strftime("%Y/%m/%d/H%H00"))
        start += timedelta(hours=1)
    return dates
