import cv2
import cv2
import numpy as np

if __name__ == '__main__':




    def get_majority_mask(masks):
        # Compute the majority mask based on the input list of masks
        # by summing them element-wise and thresholding the result
        summed_mask = sum(masks)
        majority_mask = np.where(summed_mask >= (len(masks) // 2 + 1), 1, 0).astype('uint8')
        return majority_mask


    # This function applies GrabCut to a frame using the provided mask
    def apply_grabcut(frame, mask, rect):
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        cv2.grabCut(frame, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

        mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        frame = frame * mask[:, :, np.newaxis]

        return frame, mask


    # Load the video
    cap = cv2.VideoCapture('burned2.mp4')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the output codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))

    # Variables to store the previous frames and masks
    previous_frames = []
    previous_masks = []

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Create a mask with the same shape as the frame
        mask = np.zeros(frame.shape[:2], np.uint8)

        # Define the area of interest (the rectangle around the object to keep)
        rect = (120, 30, 350, 250)

        # Apply GrabCut to the current frame
        frame, mask = apply_grabcut(frame, mask, rect)

        # Add the current frame and mask to the previous frames and masks lists
        previous_frames.append(frame)
        previous_masks.append(mask)

        # Keep only the 3 most recent frames and masks
        if len(previous_frames) > 3:
            previous_frames.pop(0)
            previous_masks.pop(0)

        # Check if we have at least 2 previous masks
        if len(previous_masks) >= 2:
            # Get the majority mask from the previous masks
            majority_mask = get_majority_mask(previous_masks)

            # Apply the majority mask to the current frame
            frame = frame * majority_mask[:, :, np.newaxis]

        # Write the frame to the output video
        out.write(frame)

        # Display the result
        cv2.imshow('Result', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
