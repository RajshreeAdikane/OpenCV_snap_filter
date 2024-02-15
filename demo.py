import numpy as np
import cv2
from keras.models import load_model

# Load the trained model
model = load_model("saved_model")

# Get frontal face haar cascade
face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

# Get webcam
camera = cv2.VideoCapture(0)

# Define constants for button locations and colors
BUTTON_1 = (10, 10, 150, 50)  # Button 1 (for filter 1)
BUTTON_2 = (170, 10, 150, 50)  # Button 2 (for filter 2)
BUTTON_3 = (330, 10, 150, 50)  # Button 3 (for filter 3)
BUTTON_COLOR = (255, 255, 255)  # Button color (white)
BUTTON_TEXT_COLOR = (0, 0, 0)  # Button text color (black)

# Define variables to track filter selection
current_filter = None

# Function to handle mouse events
def on_mouse_click(event, x, y, flags, params):
    global current_filter
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if the mouse click is within any of the button regions
        if BUTTON_1[0] <= x <= BUTTON_1[0] + BUTTON_1[2] and BUTTON_1[1] <= y <= BUTTON_1[1] + BUTTON_1[3]:
            current_filter = 1
            print("Filter 1 selected")
        elif BUTTON_2[0] <= x <= BUTTON_2[0] + BUTTON_2[2] and BUTTON_2[1] <= y <= BUTTON_2[1] + BUTTON_2[3]:
            current_filter = 2
            print("Filter 2 selected")
        elif BUTTON_3[0] <= x <= BUTTON_3[0] + BUTTON_3[2] and BUTTON_3[1] <= y <= BUTTON_3[1] + BUTTON_3[3]:
            current_filter = 3
            print("Filter 3 selected")

# Run the program infinitely
while True:
    grab_trueorfalse, img = camera.read()  # Read data from the webcam
    
    # Check if the named window exists
    if not cv2.getWindowProperty('Webcam', cv2.WND_PROP_VISIBLE) < 1:
        cv2.namedWindow('Webcam')
        cv2.setMouseCallback('Webcam', on_mouse_click)

    # Preprocess input frame from webcam
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert RGB data to Grayscale
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # Identify faces in the webcam

    # For each detected face using the Haar cascade
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        img_copy = np.copy(img)
        img_copy_1 = np.copy(img)
        roi_color = img_copy_1[y:y + h, x:x + w]

        width_original = roi_gray.shape[1]  # Width of region where face is detected
        height_original = roi_gray.shape[0]  # Height of region where face is detected
        img_gray = cv2.resize(roi_gray, (96, 96))  # Resize image to size 96x96
        img_gray = img_gray / 255  # Normalize the image data

        img_model = np.reshape(img_gray, (1, 96, 96, 1))  # Model takes input of shape = [batch_size, height, width, no. of channels]
        keypoints = model.predict(img_model)[0]  # Predict keypoints for the current input

        # Keypoints are saved as (x1, y1, x2, y2, ......)
        x_coords = keypoints[0::2]  # Read alternate elements starting from index 0
        y_coords = keypoints[1::2]  # Read alternate elements starting from index 1

        x_coords_denormalized = (x_coords + 0.5) * width_original  # Denormalize x-coordinate
        y_coords_denormalized = (y_coords + 0.5) * height_original  # Denormalize y-coordinate

        for i in range(len(x_coords)):  # Plot the keypoints at the x and y coordinates
            cv2.circle(roi_color, (int(x_coords_denormalized[i]), int(y_coords_denormalized[i])), 2, (255, 255, 0), -1)

        # Particular keypoints for scaling and positioning of the filter
        left_lip_coords = (int(x_coords_denormalized[11]), int(y_coords_denormalized[11]))
        right_lip_coords = (int(x_coords_denormalized[12]), int(y_coords_denormalized[12]))
        top_lip_coords = (int(x_coords_denormalized[13]), int(y_coords_denormalized[13]))
        bottom_lip_coords = (int(x_coords_denormalized[14]), int(y_coords_denormalized[14]))
        left_eye_coords = (int(x_coords_denormalized[3]), int(y_coords_denormalized[3]))
        right_eye_coords = (int(x_coords_denormalized[5]), int(y_coords_denormalized[5]))
        brow_coords = (int(x_coords_denormalized[6]), int(y_coords_denormalized[6]))

        # Scale filter according to keypoint coordinates
        beard_width = right_lip_coords[0] - left_lip_coords[0]
        glasses_width = right_eye_coords[0] - left_eye_coords[0]

        # Convert img_copy to BGRA format
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2BGRA)

        if current_filter == 1:
            # Beard filter
            santa_filter = cv2.imread('filters/santa_filter.png', -1)
            santa_filter = cv2.resize(santa_filter, (beard_width * 3, 150))
            sw, sh, sc = santa_filter.shape

            for i in range(0, sw):  # Overlay the filter based on the alpha channel
                for j in range(0, sh):
                    if santa_filter[i, j][3] != 0:
                        if (top_lip_coords[1] + i + y - 20) < img_copy.shape[0] and (left_lip_coords[0] + j + x - 60) < img_copy.shape[1]:
                            img_copy[top_lip_coords[1] + i + y - 20, left_lip_coords[0] + j + x - 60] = santa_filter[i, j]

        elif current_filter == 2:
            # Hat filter
            hat = cv2.imread('filters/hat2.png', -1)
            hat = cv2.resize(hat, (w, w))
            hw, hh, hc = hat.shape

            for i in range(0, hw):  # Overlay the filter based on the alpha channel
                for j in range(0, hh):
                    if hat[i, j][3] != 0:
                        img_copy[i + y - brow_coords[1] * 2, j + x - left_eye_coords[0] * 1 + 20] = hat[i, j]

        elif current_filter == 3:
            # Glasses filter
            glasses = cv2.imread('filters/glasses.png', -1)
            glasses = cv2.resize(glasses, (glasses_width * 2, 150))
            gw, gh, gc = glasses.shape

            for i in range(0, gw):  # Overlay the filter based on the alpha channel
                for j in range(0, gh):
                    if glasses[i, j][3] != 0:
                        img_copy[brow_coords[1] + i + y - 50, left_eye_coords[0] + j + x - 60] = glasses[i, j]

        # Convert img_copy back to BGR format
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGRA2BGR)

        cv2.imshow('Output', img_copy)  # Output with the filter placed on the face
        cv2.imshow('Keypoints predicted', img_copy_1)  # Place keypoints on the webcam input

    # Draw styled buttons on the webcam feed
    cv2.rectangle(img, BUTTON_1[:2], (BUTTON_1[0] + BUTTON_1[2], BUTTON_1[1] + BUTTON_1[3]), BUTTON_COLOR, -1)
    cv2.putText(img, 'Filter 1', (BUTTON_1[0] + 20, BUTTON_1[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, BUTTON_TEXT_COLOR, 2,
                cv2.LINE_AA)
    cv2.rectangle(img, BUTTON_2[:2], (BUTTON_2[0] + BUTTON_2[2], BUTTON_2[1] + BUTTON_2[3]), BUTTON_COLOR, -1)
    cv2.putText(img, 'Filter 2', (BUTTON_2[0] + 20, BUTTON_2[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, BUTTON_TEXT_COLOR, 2,
                cv2.LINE_AA)
    cv2.rectangle(img, BUTTON_3[:2], (BUTTON_3[0] + BUTTON_3[2], BUTTON_3[1] + BUTTON_3[3]), BUTTON_COLOR, -1)
    cv2.putText(img, 'Filter 3', (BUTTON_3[0] + 20, BUTTON_3[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, BUTTON_TEXT_COLOR, 2,
                cv2.LINE_AA)

    cv2.imshow('Webcam', img)  # Original webcam Input

    if cv2.waitKey(1) & 0xFF == ord("e"):  # If 'e' is pressed, stop reading and break the loop
        break

camera.release()
cv2.destroyAllWindows()
