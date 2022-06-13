"""my_dataset dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf
import os
import cv2
from pathlib import Path
import numpy as np
import random

_IMAGE = 'image'
_MASK = 'mask'
_TEST = 'test'
_TRAIN = 'train'

# TODO(my_dataset): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(my_dataset): BibTeX citation
_CITATION = """
"""


class MyDataset(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for my_dataset dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(my_dataset): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            _IMAGE: tfds.features.Image(shape=(None, None, 3), dtype=tf.uint8),
            _MASK: tfds.features.Image(shape=(None, None, 1), dtype=tf.uint8),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=(_IMAGE, _MASK),  # Set to `None` to disable
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(my_dataset): Downloads the data and defines the splits
    path = self._from_data('src')

    return {
        _TRAIN: self._generate_examples(path / _TRAIN),
        _TEST: self._generate_examples(path / _TEST)
    }


  def _generate_examples(self, path):
    image_path = path / _IMAGE
    mask_path = path / _MASK
    for image in image_path.glob('*.png'):
      mask = mask_path / image.name
      yield image.name, {
          _IMAGE: image,
          _MASK: mask,
      }
  
  def _from_data(self, path):
    find_images = lambda path: map(lambda path: (cv2.imread(str(path)), path.name), path.glob('*.jpg'))

    src_path = Path(path)
    samples = {}
    image_path = src_path / _IMAGE
    for image, filename in find_images(image_path):
      samples[filename] = {_IMAGE: image}

    mask_path = src_path / _MASK
    for mask, filename in find_images(mask_path):
      mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
      kernel = np.ones((5, 5), 'uint8')
      mask = cv2.erode(mask, kernel, iterations=1)
      inner = cv2.erode(mask, kernel, iterations=2)
      mask[np.where(mask > 0)] = 2
      mask[np.where(inner > 0)] = 1
      mask[np.where(mask == 0)] = 3
      samples[filename][_MASK] = mask

    dst_path = Path('dst')
    image_path = dst_path / _IMAGE
    mask_path = dst_path / _MASK
    try:
      os.mkdir(dst_path)
    except:
      pass
    border = int(0.8 * len(samples))
    for i in range(len(samples)):
      filename = list(samples.keys())[i]
      sample = samples[filename]
      cat_path = dst_path / (_TRAIN if i < border else _TEST)
      try:
        os.mkdir(cat_path)
      except:
        pass
      image_path = cat_path / _IMAGE
      mask_path = cat_path / _MASK
      try:
        os.mkdir(image_path)
      except:
        pass
      try:
        os.mkdir(mask_path)
      except:
        pass
      filename = filename.split('.')[0] + '.png'
      cv2.imwrite(str(image_path / filename), sample[_IMAGE])
      cv2.imwrite(str(mask_path / filename), sample[_MASK])
    return dst_path
