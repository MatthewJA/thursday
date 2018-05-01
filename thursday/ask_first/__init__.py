"""Query images from Faint Images of the Radio Sky at Twenty-Centimeters.

Matthew Alger <matthew.alger@anu.edu.au>
Research School of Astronomy and Astrophysics
The Australian National University
2018
"""

from __future__ import print_function, division

import argparse
import collections
import json
import logging
import os
import subprocess
import tempfile
import warnings

import astropy.coordinates
import astropy.io.ascii
import astropy.io.fits
import astropy.wcs
import numpy
import pandas
import scipy.spatial.distance

logger = logging.getLogger(__name__)


def make_image_table(first_path, image_table_path):
    """Generates an image metadata table if it doesn't exist.

    Parameters
    ----------
    first_path : str
        Path to FIRST data.

    image_table_path : str
        Path to write image metadata table.
    """
    subprocess.run([
        'mImgtbl',
        '-r',  # recursive
        '-c',  # corners
        first_path,
        image_table_path,
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def get_image(coord, width, first_path, image_table_path):
    """Get an image from FIRST at a coordinate.

    Parameters
    ----------
    coord : (float, float) | str
        Centre of image (RA, dec) or HHMMSS.S-DDMMSS.S string.

    width : float
        Width in degrees.

    first_path : str
        Path to FIRST data.

    image_table_path : str
        Path to image metadata table. Will be created if it doesn't
        already exist.

    Returns
    -------
    numpy.ndarray
    """
    if isinstance(coord, str):
        coord = astropy.coordinates.SkyCoord(coord, unit=('hour', 'deg'))
        coord = (coord.ra.deg, coord.dec.deg)

    make_image_table(first_path, image_table_path)

    # Get coverage information.
    with tempfile.NamedTemporaryFile() as coverage_file, \
            tempfile.NamedTemporaryFile() as out_file:
            # tempfile.NamedTemporaryFile() as header_file, \
        coverage_filename = coverage_file.name
        out_filename = out_file.name
        # header_filename = header_file.name

        subprocess.run([
            # For some reason mCoverageCheck fails unless it has two
            # initial dummy arguments. These can be pretty much anything
            # except -s and probably some other stuff but I've gone with 0
            # and 0.
            'mCoverageCheck', '0', '0',
            image_table_path, coverage_filename, '-box',
            str(coord[0]), str(coord[1]), str(width)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Confirm there's only one image containing this object
        # (i.e. no mosaic required).
        _coverage_data = coverage_file.read()
        assert _coverage_data.count(b'\n') == 4, _coverage_data

        # Get the source filename.
        coverage_file.seek(0)
        in_filename = astropy.io.ascii.read(coverage_file)['fname'][0]

        # Generate a cutout.
        subprocess.run([
            'mSubimage',
            os.path.join(first_path, in_filename),
            out_filename,
            str(coord[0]), str(coord[1]), str(width)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        return astropy.io.fits.getdata(out_filename)

        # TODO(MatthewJA): Add mosaicing.
        # Here's the outline:
        # # Coverage information is now stored in the coverage file.
        # # Mosaic the images. First we make a FITS header.
        # subprocess.run([
        #     'mMakeHdr',
        #     coverage_filename,
        #     header_filename])

        # # Then we can mosaic the images with this header as a template.
        # subprocess.run([
        #     'mAdd',
        #     '-p', first_path,
        #     '-a', 'mean',
        #     coverage_filename,
        #     header_filename,
        #     out_filename])

        # ...But this segfaults.


def read_catalogue(catalogue_path):
    """Read the FIRST catalogue.

    Parameters
    ----------
    catalogue_path : str
        Path to FIRST catalogue (decompressed ASCII file).

    Returns
    -------
    DataFrame
        Pandas dataframe of catalogue.
    """
    cat = pandas.read_fwf(
        catalogue_path, widths=[
            12, 13, 6, 9, 10, 8, 7, 7, 6, 7, 7,
            6, 13, 3, 6, 6, 2, 3, 6, 6, 9, 10, 10],
        header=1, skiprows=0)
    cat.rename(columns={
        '#  RA': 'RA',
        '#': 'SDSS #',
        'Sep': 'SDSS Sep',
        'i': 'SDSS i',
        'Cl': 'SDSS Cl',
        '#.1': '2MASS #',
        'Sep.1': '2MASS Sep',
        'K': '2MASS K',
        'Mean-yr': 'Epoch Mean-yr',
        'Mean-MJD': 'Epoch Mean-MJD',
        'rms-MJD': 'Epoch rms-MJD',
    }, inplace=True)
    return cat
