import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
POSE_CONNECTIONS_WITHOUT_HEAD = frozenset([(11, 12), (11, 13), (13, 15), (12, 14), (14, 16)])
# (b g r)
MASK_COLOR = (0, 0, 0)
BACKGROUND_COLOR = (100, 35, 1)
ANNOTATION_COLOR = (255, 255, 255)

def run_solution(image, holistic):
    # Get image and landmarks
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)

    # Draw landmark annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        mp_drawing.DrawingSpec(color=ANNOTATION_COLOR, thickness=0, circle_radius=0), # Dots
        mp_drawing.DrawingSpec(color=ANNOTATION_COLOR, thickness=2, circle_radius=1)  # Lines
    )
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        POSE_CONNECTIONS_WITHOUT_HEAD,
        mp_drawing.DrawingSpec(color=ANNOTATION_COLOR, thickness=0, circle_radius=0), # Dots
        mp_drawing.DrawingSpec(color=ANNOTATION_COLOR, thickness=1, circle_radius=0) # Lines
    )
    mp_drawing.draw_landmarks(
        image,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=ANNOTATION_COLOR, thickness=2, circle_radius=3), # Dots
        mp_drawing.DrawingSpec(color=ANNOTATION_COLOR, thickness=3, circle_radius=2)  # Lines
    )
    mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=ANNOTATION_COLOR, thickness=2, circle_radius=3), # Dots
        mp_drawing.DrawingSpec(color=ANNOTATION_COLOR, thickness=3, circle_radius=2)  # Lines
    )

def draw_silohouette(image, results):		
    fg_image = None
    bg_image = None

    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1

    # Silohouette person
    if fg_image is None:
        fg_image = np.zeros(image.shape, dtype=np.uint8)
        fg_image[:] = MASK_COLOR
    # Background
    if bg_image is None:
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = BACKGROUND_COLOR
    output_image = np.where(condition, fg_image, bg_image)
        
    return output_image
