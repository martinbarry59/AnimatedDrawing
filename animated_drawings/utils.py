# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from PIL import Image, ImageOps
import numpy as np
import numpy.typing as npt
import cv2
from pathlib import Path
import logging
from pkg_resources import resource_filename
from typing import List, Dict

TOLERANCE = 10**-5

def scale_and_pad(frame, target_w, target_h):
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    pad_left = (target_w - new_w) // 2
    pad_top = (target_h - new_h) // 2

    # Create black background and paste resized image
    result = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    result[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    return result

def overlay_with_mask(frame, character_img, x, y):
    ## translate character image to the left of its own image
    character_img = np.roll(character_img, -x, axis=1)
    roi = frame
    
    ## resize mask to fit the character image
    # Composite: if mask == 1, take from character_img; else, keep roi
    composite = np.where(1-( 1 * (character_img == 255) + 1*(character_img == 0)), character_img, roi)
    frame = composite
    return frame




def compute_orientations(joint_positions: dict, skeleton_cfg: List[dict]) -> Dict[str, float]:
    orientations = {}
    for joint in skeleton_cfg:
        name = joint['name']
        parent = joint['parent']
        
        if not parent or name not in joint_positions or parent not in joint_positions:
            continue
        p1 = np.array(joint_positions[parent])
        p2 = np.array(joint_positions[name])
        vec = p2 - p1
        angle = np.degrees(np.arctan2(vec[1], vec[0]))  # angle from +X axis
        ## Normalize angle to [0, 360)
        # if angle < 0:
        #     angle += 360
        orientations[name] = angle
    return orientations
def resolve_ad_filepath(file_name: str, file_type: str) -> Path:
    """
    Given input filename, attempts to find the file, first by relative to cwd,
    then by absolute, the relative to animated_drawings root directory.
    If not found, prints error message indicating which file_type it is.
    """
    if Path(file_name).exists():
        return Path(file_name)
    elif Path.joinpath(Path.cwd(), file_name).exists():
        return Path.joinpath(Path.cwd(), file_name)
    elif Path(resource_filename(__name__, file_name)).exists():
        return Path(resource_filename(__name__, file_name))
    elif Path(resource_filename(__name__, str(Path('..', file_name)))):
        return Path(resource_filename(__name__, str(Path('..', file_name))))

    msg = f'Could not find the {file_type} specified: {file_name}'
    logging.critical(msg)
    assert False, msg


def read_background_image(file_name: str) -> npt.NDArray[np.uint8]:
    """
    Given path to input image file, opens it, flips it based on EXIF tags, if present, and returns image with proper orientation.
    """
    # Check the file path
    file_path = resolve_ad_filepath(file_name, 'background_image')

    # Open the image and rotate as needed depending upon exif tag
    image = Image.open(str(file_path))
    image = ImageOps.exif_transpose(image)

    # Convert to numpy array and flip rightside up
    image_np = np.asarray(image)
    image_np = cv2.flip(image_np, 0)

    # Ensure we have RGBA
    if len(image_np.shape) == 3 and image_np.shape[-1] == 3:  # if RGB
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2RGBA)
    if len(image_np.shape) == 2:  # if grayscale
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGBA)

    return image_np.astype(np.uint8)
