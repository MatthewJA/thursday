import io
import requests
import urllib

from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier
import astropy.units as u
import h5py
import numpy as np
import pandas
import skimage.transform

import ask_first
from second.first_download import download_first
from second import get_data

Vizier.ROW_LIMIT = -1

def create_labels(astropy_table, raw_labels):
   """Create labels from astopy table..
   
   # Arguments
      astropy_table: Astropy table from Vizier 
         region query
      raw_labels: Array of original labels.

   # Returns
      Array of labels for components. 
   """

   unique, indices, counts = np.unique(astropy_table['_q'], 
                                       return_index=True,  
                                       return_counts=True)
   samples = len(astropy_table['_q'])
   labels = np.zeros((samples,))

   for i, value in enumerate(unique):
      index = indices[i]
      label = raw_labels[value]
      nb = counts[i]
      labels[index:index + nb] = np.full(nb, label)

   labels = labels.reshape(-1,)
   return labels


def download_fr_components(output_path: str):
   """Downloads FR component cutouts.
   
   # Arguments
      output_path: File path to save h5py file.
   """
   # URL shortened using bitly.com
   r = requests.get('https://go.nasa.gov/2jlAMfN')

   fricat = pandas.read_csv(io.StringIO(r.text), sep='|', skiprows=4)
   fricat.columns = fricat.columns.str.strip()

   friicat = Vizier.get_catalogs('J/A+A/601/A81')[0].to_pandas()

   assert len(fricat) == 233
   assert len(friicat) == 123

   fri_labels = np.full(len(fricat), 1)
   frii_labels = np.full(len(friicat), 2)

   source_labels = np.concatenate((fri_labels, frii_labels), axis=0)
      
   # Coverting coordinate of varying formats to SkyCoords 
   fri_coord = SkyCoord(fricat['ra'], 
                        fricat['dec'],
                        unit=('hour', 'deg'))
   frii_coord = SkyCoord(friicat['_RA'], 
                        friicat['_DE'],
                        unit=('deg', 'deg'))

   while True:
      try:
         fri_table = Vizier.query_region(fri_coord, 
                                 radius= 2.35 * u.arcmin, 
                                 catalog='VIII/92')[0]
      
         frii_table = Vizier.query_region(frii_coord, 
                                 radius= 2.68 * u.arcmin, 
                                 catalog='VIII/92')[0]
         break

      except:
         print ("Error: Failed to connect to server. " + 
                "Reconnecting...")


   print ("FR1  " + str(fri_table.to_pandas().shape[0]))
   print ("FR2 " + str(frii_table.to_pandas().shape[0]))

   fri_c = SkyCoord(ra=fri_table['RAJ2000'], 
                    dec=fri_table['DEJ2000'], 
                    unit=('hourangle', 'deg'))

   frii_c = SkyCoord(ra=frii_table['RAJ2000'], 
                    dec=frii_table['DEJ2000'], 
                    unit=('hourangle', 'deg'))
         
   # Creating labels
   fri_labels = create_labels(fri_table, source_labels)
   frii_labels = create_labels(frii_table, source_labels)
   labels = np.concatenaten((fri_labels, frii_labels), axis=0)

   fri_images = np.zeros((fri_c.shape[0], 300, 300))
   frii_images = np.zeros((frii_c.shape[0], 300, 300))


   for i in range(fri_c.shape[0]):
      coord = fri_c[i]
      im = get_data.first(coord, size=10)
      im = im[0].data

      im -= im.min()
      im /= im.max()
      im = skimage.transform.resize(im, (300, 300))

      print ("Downloaded " + str(i) + "/" + str(fri_c.shape[0]) + " FRI")

      fri_images[i, :, :] = im

      
   for i in range(frii_c.shape[0]):
      coord = frii_c[i]
      im = get_data.first(coord, size=10)
      im = im[0].data

      im -= im.min()
      im /= im.max()
      im = skimage.transform.resize(im, (300, 300))

      print ("Downloaded " + str(i) + "/" + str(frii_c.shape[0]) + " FRII")

      frii_images[i, :, :] = im

   images = np.vstack((fri_images, frii_images))

   with h5py.File(first_data_path, 'r+') as f:
      f.create_dataset('images', data=images)
      f.create_dataset('labels', data=labels)
      f.create_dataset('fri_data', 
                        data=(fri_c.ra.hourangle, 
                              fri_c.dec.deg))
      f.create_dataset('frii_data', 
                        data=(frii_c.ra.deg, 
                              frii_c.dec.deg))


def download_random(output_path: str, n=1000):
   """Download random radio cutouts from FIRST.
   
   # Arguments
      output_path: File path to save h5py file
   """
   Vizier.ROW_LIMIT = -1
   catalog = Vizier.get_catalogs(catalog='VIII/92')[0]
   table = catalog.to_pandas()

   indices = list(range(len(table)))
   np.random.shuffle(indices)
   selection = indices[:n]

   selected = table.iloc[selection]
   images = numpy.zeros((n, 300, 300))

   coords = SkyCoord(ra=selected['RAJ2000'], 
                    dec=selected['DEJ2000'], 
                    unit=('hourangle', 'deg'))
      
   print ("Downloading Random Images")
   for i in range(coords.shape[0]):
      coord = coords[i]
      im = get_data.first(coord, size=10)
      im = im[0].data

      im -= im.min()
      im /= im.max()
      im = skimage.transform.resize(im, (300, 300))

      print ("Downloaded " + str(i) + "/" + str(n) + " Random Images")

      images[i, :, :] = im
      

   with h5py.File(output_path, 'r+') as f:
      f.create_dataset('images', data=images)
      f.create_dataset('data',
                        data=(coords.ra.hourangle, 
                              coords.dec.deg))
