import cv2
import mediapipe as mp
import json
import yaml
# List of Mediapipe Pose landmark names
from typing import List, Dict
import numpy as np
from animated_drawings.config import Config, CharacterConfig, RetargetConfig, MotionConfig
from animated_drawings.model_live.animated_drawing import AnimatedDrawing
from typing import List, Tuple, Dict
from OpenGL import GL
cfg: Config = Config('examples/config/mvc/rockandrollarmy.yaml')
# create view
# from animated_drawings.view.view import View
# view = View.create_view(cfg.view)

from animated_drawings.model_live.scene import Scene
from animated_drawings.view.view import View
from animated_drawings.model_live.scene import Scene
scene = Scene(cfg.scene)

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
view = View.create_view(cfg.view)
view.render(scene)
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
LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_hand", "right_hand", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_foot", "right_foot", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index"
]

VIRTUAL_JOINTS = {
    "neck": ["left_shoulder", "right_shoulder", "nose"],  # neck = avg of shoulders + nose
    "torso": ["left_shoulder", "right_shoulder"],
    "hip": ["right_hip","left_hip"],
    "root": ["hip"],  # root = avg of torso and hip
}

with open("examples/characters/robot/char_cfg.yaml", "r") as f:
    rig_data = yaml.safe_load(f)
    limbs = rig_data.get("skeleton", [])
    width = rig_data['width']
    height = rig_data['height']

name_to_index = {name: idx for idx, name in enumerate(LANDMARK_NAMES)}
# Mediapipe setup
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

cap = cv2.VideoCapture(0)
width = 640
height = 480
frame_data = np.empty([height, width, 4], dtype='uint8')  # 4 for RGBA
i = 0
try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue
        # Convert frame for Mediapipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        height, width, _ = frame.shape

        named_landmarks = {}

        if results.pose_landmarks:
            
            json.dumps(named_landmarks, indent=2)
            landmark_dict = {}
            for name, idx in name_to_index.items():
                landmark = results.pose_landmarks.landmark[idx]
                x, y = int(landmark.x * width), int(landmark.y * height)
                landmark_dict[name] = (x,y)
            
            adjusted_landmarks = {
                name: (x, height - y)
                # name: (y,x)  # Adjust y-coordinate to match OpenGL's coordinate system
                for name, (x, y) in landmark_dict.items()
            }
            for virtual_name, source_joints in VIRTUAL_JOINTS.items():
                if all(j in landmark_dict for j in source_joints):
                    xs = [adjusted_landmarks[j][0] for j in source_joints]
                    ys = [adjusted_landmarks[j][1] for j in source_joints]
                    cx, cy = int(sum(xs) / len(xs)), int(sum(ys) / len(ys))
                    adjusted_landmarks[virtual_name] = (cx, cy)
                    xs = [landmark_dict[j][0] for j in source_joints]
                    ys = [landmark_dict[j][1] for j in source_joints]
                    cx, cy = int(sum(xs) / len(xs)), int(sum(ys) / len(ys))
                    landmark_dict[virtual_name] = (cx, cy)
            # Draw lines for each limb from parent to child
            for limb in limbs:
                name = limb["name"]
                parent = limb["parent"]
                if name in landmark_dict and parent in landmark_dict:
                    p1 = landmark_dict[parent]
                    p2 = landmark_dict[name]
                    cv2.line(frame, p1, p2, (0, 255, 0), 3)
                    cv2.circle(frame, p2, 5, (0, 0, 255), -1)
                    cv2.putText(frame, name, p2, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            
            angles = compute_orientations(adjusted_landmarks, limbs)
            view.clear_window()
            scene.update_transforms()
            
            
            for child in scene.get_children():
                child.live_angles = angles
                child.live_root_position = np.array([adjusted_landmarks["root"][0], adjusted_landmarks["root"][1], 0])
                # child.rig.root_joint.set_position(root_position)
                # child.rig.set_global_orientations(angles)
                # child.update()
            view.render(scene)
            scene.progress_time(1/30)
            GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, 0)
            # GL.glReadPixels(0, 0, width, height, GL.GL_BGRA, GL.GL_UNSIGNED_BYTE, frame_data)
            # GL.glFinish()  # ensure all GL rendering is done before reading
            # image = frame_data[::-1, :, :].copy()
            # w, h = scene.get_children()[0].img_dim, scene.get_children()[0].img_dim
            raw = GL.glReadPixels(0, 0, width, height, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
            image = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 4))
            ## mirror image
            image = cv2.flip(image, 0)
            # # Read pixels from OpenGL framebuffer
            # buffer = GL.glReadPixels(0, 0, w, h, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
            # Convert to NumPy array
            char_image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB).astype(np.uint8)
            char_image =  np.rot90(char_image,3)
            char_image = cv2.resize(char_image, (width, height), interpolation=cv2.INTER_LINEAR)
            ## fit image to frame
            

            
            frame = overlay_with_mask(frame, char_image , x=200, y=0)  # top-left corner
        
        scaled_frame = scale_and_pad(frame, 2 * width, 2 * height)

        cv2.imshow("Mediapipe Pose with Named Landmarks", scaled_frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

finally:
    
    cap.release()
    cv2.destroyAllWindows()
    pose.close()
