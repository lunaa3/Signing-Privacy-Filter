import cv2
import mediapipe as mp

import holistic_solution

mp_holistic = mp.solutions.holistic

class WebcamFilter():
    def run(self):
        # For webcam input:
        cap = cv2.VideoCapture(0)

        with mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=0,
            smooth_landmarks=True,
            enable_segmentation=True,
            smooth_segmentation=True,
            refine_face_landmarks=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3) as holistic:

            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print('Ignoring empty camera frame.')
                    continue

                # Get landmarks
                image, results = holistic_solution.run_solution(image, holistic)

                # Checking if there is a segmentation mask to prevent a crash.
                # Only draw filtered image with silohouette mask
                if (results.segmentation_mask is not None):
                    # Get image with silohouette mask
                    results.segmentation_mask = results.segmentation_mask
                    masked_image = holistic_solution.draw_silohouette(image, results)
                    # Draw holistic landmarks on the masked image
                    holistic_solution.draw_landmarks(masked_image, results)

                    # Flip the image horizontally for a selfie-view display.
                    cv2.imshow('MediaPipe Holistic', cv2.flip(masked_image, 1))

                # Break gracefully
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()
