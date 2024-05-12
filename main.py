import cv2
import numpy as np
import tensorflow as tf

# Load your pre-trained gesture detection model
model = tf.keras.models.load_model('adam.h5')

# Initialize the camera
cap = cv2.VideoCapture(0)

# Define the coordinates of the ROI (region of interest)
x, y, w, h = 100, 100, 200, 200  # Example: select a square region at (100, 100) with width and height of 200 pixels

def prediction(pred):
    return chr(pred + 65)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Extract the region of interest (ROI)
    roi = gray_frame[y:y+h, x:x+w]
    
    # Preprocess the ROI if needed (resize, normalize, etc.)
    # For example, resizing to 28x28 if your model expects that size
    roi = cv2.resize(roi, (28, 28))
    
    # Perform gesture detection
    # For example, if using a deep learning model
    prediction_result = model.predict(np.expand_dims(roi, axis=0))
    pred_class = np.argmax(prediction_result)
    print(prediction_result)
    
    # Get the predicted character
    predicted_char = prediction(pred_class)
    
    # Display the ROI with a bounding box
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    frame = cv2.flip(frame, 1)
    
    # Display the recognized gesture
    cv2.putText(frame, predicted_char, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    
    # Flip the frame horizontally
 
    # Display the resulting frame
    cv2.imshow('Gesture Detection', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

