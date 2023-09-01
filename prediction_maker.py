import cv2
import numpy as np
from tensorflow import keras

# this function will load the trained model
def load_model():
    model = keras.models.load_model('/Users/isaackim/Python/simple_neural_network/trained_model.h5')
    return model

def classify_image(model, image):
    # process the image
    prediction = model.predict(np.expand_dims(image, axis=0))[0]
    predicted_label = np.argmax(prediction)
    # arg max bc u want to return the output w the highest prob
    return predicted_label

model = load_model()



#open webcam
cap = cv2.VideoCapture(0)

# this loop continuously captures frames from hte webcam, preprocesses them, amkes predictions and displays hte result
# loop keeps running until the user presses the 'q' key
# this is for when u want to process to run indefinitely

while True:
    # capture a frame
    ret, frame = cap.read()

    #preprocessing
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray_frame, (28, 28))
    normalized_frame = resized_frame/255.0
    predicted_label = classify_image(model, normalized_frame)

    # put the prediction on hte frame
    cv2.putText(frame, f"Predicted Label: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    #display the frame
    cv2.imshow('Webca, Feed', frame)

    # exit the loop if hte 'q' is pressed
    # wait key function is for waiting until stop display i think
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
