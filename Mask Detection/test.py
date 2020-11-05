import cv2
import os
import numpy as np
from PIL import Image
import torch
from utils import DNN, get_test_loader


proto_path = "models/deploy.prototxt.txt"
model_path = "models/res10_300x300_ssd_iter_140000.caffemodel"

# Face detection opencv pretrained model to detect faces in images
fd_net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

# Transform to apply on test images
test_transform = get_test_loader()

# Initializing mask detection model
mask_detector = DNN()

# Loading it's weights from file
mask_detector.load_state_dict(torch.load('models/Mobilenet_v2_Mask_Detection.pt', map_location = torch.device('cpu')))

# Switch model to inference mode
mask_detector.eval()

# img_path = input('Enter image path: ')
img_path = 'images/sample3.jpg'

# Read input image
image = cv2.imread(img_path)

image_copy = image.copy()
(h, w) = image.shape[:2]

# Get detected faces
blob = cv2.dnn.blobFromImage(cv2.resize(image_copy, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
fd_net.setInput(blob)
detections = fd_net.forward()

# Iterate through detected faces and run mask detection model
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > 0.8:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype("int")

        # Extract rectangle (containing face) from input image and convert it to PIL image
        face = cv2.cvtColor(image_copy[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        face = Image.fromarray(face)
        # Apply test transform to feed it to the model
        face_tensor = test_transform(face).unsqueeze(0)

        # Run forward pass
        out = mask_detector(face_tensor).item()

        # Apply sigmoid to get probability
        prob = 1/(1 + np.exp(-out))

        # Draw rectangle around face in the image
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # If prob is > .5 predict person's wearing mask else not
        if prob > 0.5:
            cv2.putText(image, "Mask off: {:.2f}".format(1-prob), (x1, y1), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
        else:
            cv2.putText(image, "Mask on: {:.2f}".format(1-prob), (x1, y1), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0))

        cv2.imwrite(img_path.split('.')[0] + '_out.jpg', image)
        print('Output image saved')
