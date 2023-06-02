import cv2
import mediapipe as mp
from os.path import basename, splitext

import holistic_solution

mp_holistic = mp.solutions.holistic

class VideoFilter():   
    def run(self, file_name):
        # For video input:
        cap = cv2.VideoCapture(file_name)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        print(f'Resolution: {width}x{height}')
        print(f'Fps: {fps}')

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(splitext(basename(file_name))[0] + '_filtered.mp4', fourcc, fps, (width, height))

        with mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=0,
            smooth_landmarks=True,
            enable_segmentation=True,
            smooth_segmentation=True,
            refine_face_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as holistic:

            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print('Ignoring empty camera frame.')
                    break

                # Get landmarks
                image, results = holistic_solution.run_solution(image, holistic)

                # Checking if there is a segmentation mask to prevent a crash.
                # Only draw filtered image with silohouette mask
                if (results.segmentation_mask is not None):
                    # Get image with silohouette mask
                    masked_image = holistic_solution.draw_silohouette(image, results)
                    # Draw holistic landmarks on the masked image
                    holistic_solution.draw_landmarks(masked_image, results)

                   # Write the filtered frame
                    out.write(masked_image)

                    # Show filtered video
                    cv2.imshow('OpenCV Feed', masked_image)

                # Break gracefully
                if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            cap.release()
            out.release()
            print('done')
            cv2.destroyAllWindows()
