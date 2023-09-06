"""lanternfly dataset."""

import os
import xml.etree.ElementTree

import tensorflow as tf
import tensorflow_datasets as tfds

def _get_example_objects(annon_filepath):
  """Function to get all the objects from the annotation XML file."""
  with tf.io.gfile.GFile(annon_filepath, "r") as f:
    root = xml.etree.ElementTree.parse(f).getroot()

    size = root.find("size")
    width = float(size.find("width").text)
    height = float(size.find("height").text)

    for obj in root.findall("object"):
      label = obj.find("name").text.lower()
      bndbox = obj.find("bndbox")
      xmax = float(bndbox.find("xmax").text)
      xmin = float(bndbox.find("xmin").text)
      ymax = float(bndbox.find("ymax").text)
      ymin = float(bndbox.find("ymin").text)
      yield {
          "label": label,
          "bbox": tfds.features.BBox(
              ymin / height, xmin / width, ymax / height, xmax / width
          ),
      }

class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for lanternfly dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(),
            'image/filename': tfds.features.Text(),
            'objects': tfds.features.Sequence({
              'label': tfds.features.ClassLabel(names=['egg masses', 'instar nymph (1-3)', 'instar nymph (4)', 'adult']),
              'bbox': tfds.features.BBoxFeature()
            }),
            'labels': tfds.features.Sequence(
                tfds.features.ClassLabel(names=['egg masses', 'instar nymph (1-3)', 'instar nymph (4)', 'adult'])
            )
        })
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    path = '/home/taylor/SpottedLanternFly/keras-training/labeling'
    return {
        'train': self._generate_examples(path, 'train'),
        'test': self._generate_examples(path, 'test')
    }

  def _generate_examples(self, data_path, set_name):
    """Yields examples."""
    set_filepath = os.path.join(
        data_path,
        os.path.normpath(
            "image_sets/{}.txt".format(set_name)
        ),
    )
    
    with tf.io.gfile.GFile(set_filepath, "r") as f:
      for line in f:
        image_id = line.strip()
        example = self._generate_example(data_path, image_id)
        yield image_id, example

  def _generate_example(self, data_path, image_id):
    image_filepath = os.path.join(
        data_path,
        os.path.normpath(
            "images/{}.png".format(image_id)
        ),
    )
    annon_filepath = os.path.join(
        data_path,
        os.path.normpath(
            "annotations/{}.xml".format(image_id)
        ),
    )
    
    objects = list(_get_example_objects(annon_filepath))
    labels = sorted(set(obj["label"] for obj in objects))
    
    return {
        "image": image_filepath,
        "image/filename": image_id + ".png",
        "objects": objects,
        "labels": labels,
    }
