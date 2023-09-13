import numpy as np
import streamlit as st
import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO('yolov8n.pt')

# Define the app title and description
st.title("Animal Detection")
st.markdown("Upload an image to detect animals in it")

# Upload the image
uploaded_image = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_image:
    # Convert the uploaded file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    
    # Display the uploaded image
    st.image(opencv_image, channels="BGR", caption="Uploaded Image", use_column_width=True)
    
    # Predict animals in the image
    results = model(opencv_image)
    
    if not isinstance(results, list):
        results = [results]

    detected_labels = []
    box_id = 0
    for result in results:
        # Unpack the bounding boxes, confidence values, and classes
        boxes = result.boxes.xyxy.tolist()
        confidences = result.boxes.conf.tolist()
        classes = result.boxes.cls.tolist()
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            conf = confidences[i]
            cls = int(classes[i])
            label = model.names[cls]
            # Assign a unique identifier to each box
            box_id += 1
            # Put the box number below the box
            cv2.putText(opencv_image, str(box_id), (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Define the label_with_confidence with box_id
            label_with_confidence = f"{box_id}. {label} ({conf*100:.2f}%)"
            detected_labels.append(label_with_confidence)

            # Draw the bounding box on the image
            cv2.rectangle(opencv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Put the label with confidence on the image (this is optional, as you already have the list of labels)
            cv2.putText(opencv_image, label_with_confidence, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display the image with detected animals
    st.image(opencv_image, channels="BGR", caption="Detected Objects", use_column_width=True)
    
    # Display detected labels
    for lbl in detected_labels:
        st.write(lbl)
