# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import cv2
import numpy as np
import tensorflow as tf




# Press the green button in the gutter to run the script.
if __name__ == '__main__':


    # Load the video
    cap = cv2.VideoCapture('mug.mp4')

    # Get the frames per second (fps) of the video
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Get the width and height of the video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the output codec and create a VideoWriter object to save the result
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))

    # Loop through the frames of the video
    while (cap.isOpened()):
        ret, frame = cap.read()

        # Check if the frame was successfully read
        if not ret:
            break

        # Create a mask with the same shape as the frame
        mask = np.zeros(frame.shape[:2], np.uint8)

        # Define the background and foreground model
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        # Define the area of interest (the rectangle around the object to keep)
        rect = (50, 50, 300, 500)

        # Run the GrabCut algorithm
        cv2.grabCut(frame, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

        # Create a mask that only includes the foreground
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

        # Apply the mask to the frame to remove the background
        frame = frame * mask2[:, :, np.newaxis]

        # Write the frame to the output video
        out.write(frame)

        # Display the result
        cv2.imshow('Result', frame)

        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print("hey")
    # Release the resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
