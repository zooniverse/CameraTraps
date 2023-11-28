import io
import json
import math
import os
import sys
from datetime import datetime
from io import BytesIO
from typing import Union
from pytorch_detector import PTDetector

from PIL import Image
import numpy as np
import requests
from azure.storage.blob import ContainerClient

PRINT_EVERY = 500


#%% Helper functions *copied* from ct_utils.py and visualization/visualization_utils.py
IMAGE_ROTATIONS = {
    3: 180,
    6: 270,
    8: 90
}

def truncate_float(x, precision=3):
    """
    Function for truncating a float scalar to the defined precision.
    For example: truncate_float(0.0003214884) --> 0.000321
    This function is primarily used to achieve a certain float representation
    before exporting to JSON
    Args:
    x         (float) Scalar to truncate
    precision (int)   The number of significant digits to preserve, should be
                      greater or equal 1
    """

    assert precision > 0

    if np.isclose(x, 0):
        return 0
    else:
        # Determine the factor, which shifts the decimal point of x
        # just behind the last significant digit
        factor = math.pow(10, precision - 1 - math.floor(math.log10(abs(x))))
        # Shift decimal point by multiplicatipon with factor, flooring, and
        # division by factor
        return math.floor(x * factor)/factor


def open_image(input_file: Union[str, BytesIO]) -> Image:
    """Opens an image in binary format using PIL.Image and converts to RGB mode.

    This operation is lazy; image will not be actually loaded until the first
    operation that needs to load it (for example, resizing), so file opening
    errors can show up later.

    Args:
        input_file: str or BytesIO, either a path to an image file (anything
            that PIL can open), or an image as a stream of bytes

    Returns:
        an PIL image object in RGB mode
    """
    if (isinstance(input_file, str)
            and input_file.startswith(('http://', 'https://'))):
        response = requests.get(input_file)
        image = Image.open(BytesIO(response.content))
        try:
            response = requests.get(input_file)
            image = Image.open(BytesIO(response.content))
        except Exception as e:
            print(f'Error opening image {input_file}: {e}')
            raise
    else:
        image = Image.open(input_file)
    if image.mode not in ('RGBA', 'RGB', 'L'):
        raise AttributeError(f'Image {input_file} uses unsupported mode {image.mode}')
    if image.mode == 'RGBA' or image.mode == 'L':
        # PIL.Image.convert() returns a converted copy of this image
        image = image.convert(mode='RGB')

    # alter orientation as needed according to EXIF tag 0x112 (274) for Orientation
    # https://gist.github.com/dangtrinhnt/a577ece4cbe5364aad28
    # https://www.media.mit.edu/pia/Research/deepview/exif.html
    try:
        exif = image._getexif()
        orientation: int = exif.get(274, None)  # 274 is the key for the Orientation field
        if orientation is not None and orientation in IMAGE_ROTATIONS:
            image = image.rotate(IMAGE_ROTATIONS[orientation], expand=True)  # returns a rotated copy
    except Exception:
        pass

    return image


def load_image(input_file: Union[str, BytesIO]) -> Image.Image:
    """Loads the image at input_file as a PIL Image into memory.
    Image.open() used in open_image() is lazy and errors will occur downstream
    if not explicitly loaded.
    Args:
        input_file: str or BytesIO, either a path to an image file (anything
            that PIL can open), or an image as a stream of bytes
    Returns: PIL.Image.Image, in RGB mode
    """
    image = open_image(input_file)
    image.load()
    return image

#%% Scoring script

class BatchScorer:
    """
    Coordinates scoring images in this Task.

    1. have a synchronized queue that download tasks enqueue and scoring function dequeues - but need to be able to
    limit the size of the queue. We do not want to write the image to disk and then load it in the scoring func.
    """
    def __init__(self, **kwargs):
        print('score_v5.py BatchScorer, __init__()')

        detector_path = kwargs.get('detector_path')

        self.detector = PTDetector(detector_path)

        self.use_url = kwargs.get('use_url')
        if not self.use_url:
            input_container_sas = kwargs.get('input_container_sas')
            self.input_container_client = ContainerClient.from_container_url(input_container_sas)

        self.detection_threshold = kwargs.get('detection_threshold')

        self.image_ids_to_score = kwargs.get('image_ids_to_score')

        # determine if there is metadata attached to each image_id
        self.metadata_available = True if isinstance(self.image_ids_to_score[0], list) else False

    def _download_image(self, image_file) -> Image:
        """
        Args:
            image_file: Public URL if use_url, else the full path from container root

        Returns:
            PIL image loaded
        """
        if not self.use_url:
            downloader = self.input_container_client.download_blob(image_file)
            image_file = io.BytesIO()
            blob_props = downloader.download_to_stream(image_file)

        image = open_image(image_file)
        return image

    def score_images(self) -> list:
        detections = []

        for i in self.image_ids_to_score:

            if self.metadata_available:
                image_id = i[0]
                image_metadata = i[1]
            else:
                image_id = i

            try:
                image = self._download_image(image_id)
            except Exception as e:
                print(f'score_v5.py BatchScorer, score_images, download_image exception: {e}')
                result = {
                    'file': image_id,
                    'failure': self.detector.FAILURE_IMAGE_OPEN
                }
            else:
                result = self.detector.generate_detections_one_image(
                    image, image_id, detection_threshold=self.detection_threshold)

            if self.metadata_available:
                result['meta'] = image_metadata

            detections.append(result)
            if len(detections) % PRINT_EVERY == 0:
                print(f'scored {len(detections)} images')

        return detections


def main():
    print('score_v5.py, main()')

    # information to determine input and output locations
    api_instance_name = os.environ['API_INSTANCE_NAME']
    job_id = os.environ['AZ_BATCH_JOB_ID']
    task_id = os.environ['AZ_BATCH_TASK_ID']
    mount_point = os.environ['AZ_BATCH_NODE_MOUNTS_DIR']

    # other parameters for the task
    begin_index = int(os.environ['TASK_BEGIN_INDEX'])
    end_index = int(os.environ['TASK_END_INDEX'])

    input_container_sas = os.environ.get('JOB_CONTAINER_SAS', None)  # could be None if use_url
    use_url = os.environ.get('JOB_USE_URL', None)

    if use_url and use_url.lower() == 'true':  # bool of any non-empty string is True
        use_url = True
    else:
        use_url = False

    detection_threshold = float(os.environ['DETECTION_CONF_THRESHOLD'])

    print(f'score_v5.py, main(), api_instance_name: {api_instance_name}, job_id: {job_id}, task_id: {task_id}, '
          f'mount_point: {mount_point}, begin_index: {begin_index}, end_index: {end_index}, '
          f'input_container_sas: {input_container_sas}, use_url (parsed): {use_url}'
          f'detection_threshold: {detection_threshold}')

    job_folder_mounted = os.path.join(mount_point, 'batch-api', f'api_{api_instance_name}', f'job_{job_id}')
    task_out_dir = os.path.join(job_folder_mounted, 'task_outputs')
    os.makedirs(task_out_dir, exist_ok=True)
    task_output_path = os.path.join(task_out_dir, f'job_{job_id}_task_{task_id}.json')

    # test that we can write to output path; also in case there is no image to process
    with open(task_output_path, 'w') as f:
        json.dump([], f)

    # list images to process
    list_images_path = os.path.join(job_folder_mounted, f'{job_id}_images.json')
    with open(list_images_path) as f:
        list_images = json.load(f)
    print(f'score_v5.py, main(), length of list_images: {len(list_images)}')

    if (not isinstance(list_images, list)) or len(list_images) == 0:
        print('score_v5.py, main(), zero images in specified overall list, exiting...')
        sys.exit(0)

    # items in this list can be strings or [image_id, metadata]
    list_images = list_images[begin_index: end_index]
    if len(list_images) == 0:
        print('score_v5.py, main(), zero images in the shard, exiting')
        sys.exit(0)

    print(f'score_v5.py, main(), processing {len(list_images)} images in this Task')

    # model path
    # Path to .pb TensorFlow detector model file, relative to the
    # models/megadetector_copies folder in mounted container
    detector_model_rel_path = os.environ['DETECTOR_REL_PATH']
    detector_path = os.path.join(mount_point, 'models', 'megadetector_copies', detector_model_rel_path)
    assert os.path.exists(detector_path), f'detector is not found at the specified path: {detector_path}'

    # score the images
    scorer = BatchScorer(
        detector_path=detector_path,
        use_url=use_url,
        input_container_sas=input_container_sas,
        detection_threshold=detection_threshold,
        image_ids_to_score=list_images
    )

    try:
        tick = datetime.now()
        detections = scorer.score_images()
        duration = datetime.now() - tick
        print(f'score_v5.py, main(), score_images() duration: {duration}')
    except Exception as e:
        raise RuntimeError(f'score_v5.py, main(), exception in score_images(): {e}')

    with open(task_output_path, 'w', encoding='utf-8') as f:
        json.dump(detections, f, ensure_ascii=False)


if __name__ == '__main__':
    main()
