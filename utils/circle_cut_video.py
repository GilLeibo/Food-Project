import cv2
import numpy as np
from moviepy.editor import VideoFileClip
if __name__ == '__main__':

    def focus_on_circle(input_path, output_path, center, radius):
        # Load the video clip
        clip = VideoFileClip(input_path)

        # Define a function to blur the frame outside the circle
        def blur_frame_outside_circle(frame):
            # Create an empty mask of zeros with the same shape as the frame
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)

            # Generate a circular ROI mask with the specified center and radius
            cv2.circle(mask, center, radius, (255), -1)

            # Invert the mask to focus on the area outside the circle
            mask_inv = cv2.bitwise_not(mask)

            # Apply bitwise AND operation to frame and mask
            result_frame = cv2.bitwise_and(frame, frame, mask=mask_inv)

            return result_frame

        # Apply the blur effect to the video clip
        result_clip = clip.fl_image(blur_frame_outside_circle)

        # Write the result to the output file
        result_clip.write_videofile(output_path, codec="libx264")

        # Close the clips
        clip.close()
        result_clip.close()

    # Provide the paths to your input and output video files
    input_video_path = "Baking_Cookies_Alt.mp4"
    output_video_path = "output_video.mp4"

    # Specify the center and radius of the circle ROI
    center = (500, 200)  # Example: (x, y) coordinates of the center
    radius = 300         # Example: Radius of the circle

    # Focus on the area outside the specified circle in the video
    focus_on_circle(input_video_path, output_video_path, center, radius)
