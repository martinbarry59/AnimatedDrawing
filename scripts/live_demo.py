import cv2
import mediapipe as mp
import json
# List of Mediapipe Pose landmark names
import numpy as np
from animated_drawings.config import Config
from OpenGL import GL

# create view
# from animated_drawings.view.view import View
# view = View.create_view(cfg.view)

from animated_drawings.model_live.scene import Scene
from animated_drawings.view.view import View
from animated_drawings.model_live.scene import Scene
import os
from scripts.utils import parent_dir
import sys
from animated_drawings.utils import overlay_with_mask, scale_and_pad, compute_orientations
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
    # "neck": ["left_shoulder", "right_shoulder", "nose"],  # neck = avg of shoulders + nose
    
    "torso": ["left_shoulder", "right_shoulder"],
    "hip": ["right_hip","left_hip"],
    "root": ["hip"],  # root = avg of torso and hip
}


if __name__ == "__main__":
    yaml_path = os.path.join(parent_dir, "drawings/config/mvc/",sys.argv[1]) if len(sys.argv) > 1 else os.path.join(parent_dir, "drawings/config/mvc/", "live_demo.yaml")
    cfg: Config = Config(yaml_path)
    scene = Scene(cfg.scene)
    view = View.create_view(cfg.view)
    view.render(scene)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    limbs = scene.get_children()[0].char_cfg.skeleton
    name_to_index = {name: idx for idx, name in enumerate(LANDMARK_NAMES)}
    # Mediapipe setup
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    cap = cv2.VideoCapture(0)
    width = 640
    height = 480
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
                
                char_n = 0
                for child in scene.get_children():
                    child.live_angles = angles
                    ## fixed x,y position  start bottom left corner and so one for different characters
                    x = (char_n % 2) *  width - 150 #+ adjusted_landmarks["root"][1]
                    y = (char_n // 2 ) * 1.2 * height - 350 #+ adjusted_landmarks["root"][0]
                    char_n += 1
                    child.live_root_position = np.array([x,y, 0])

                view.render(scene)
                scene.progress_time(1/30)
                GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, 0)

                raw = GL.glReadPixels(0, 0, width, height, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
                image = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 4))
                ## mirror image
                image = cv2.flip(image, 0)

                char_image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB).astype(np.uint8)
                char_image =  np.rot90(char_image,3)
                char_image = cv2.resize(char_image, (width, height), interpolation=cv2.INTER_LINEAR)
                ## fit image to frame
                

                
                frame = overlay_with_mask(frame, char_image , x=0, y=0)  # top-left corner
            
            scaled_frame = scale_and_pad(frame, 2 * width, 2 * height)

            cv2.imshow("Mediapipe Pose with Named Landmarks", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

    finally:
        
        cap.release()
        cv2.destroyAllWindows()
        pose.close()
