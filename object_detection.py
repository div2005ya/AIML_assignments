
# Import OpenCV for reading and displaying video frames
import cv2

# Import Pandas to read the Excel annotation dataset
import pandas as pd

# Import YOLO for object detection
from ultralytics import YOLO

# Import Path to check whether files exist
from pathlib import Path


# Step 1: Define the dataset and video file paths
excel_path = "Practical_09_CCTV_Object_Detection_Annotations.xlsx"
video_path = "cctv.mp4"

# Check whether the Excel file exists
if not Path(excel_path).exists():
    raise FileNotFoundError("Excel annotation file was not found.")

# Check whether the CCTV video exists
if not Path(video_path).exists():
    raise FileNotFoundError(
        "CCTV video not found. Place cctv.mp4 in this folder "
        "or update video_path with the correct file path."
    )


# Step 2: Load and inspect the annotation dataset
annotations = pd.read_excel(excel_path)

# Display the first five annotation records
print("First five annotation records:")
print(annotations.head())

# Display the number of annotation records
print("\nTotal annotation records:", len(annotations))

# Display the number of unique images referenced
print("Unique image filenames:", annotations["Image_Filename"].nunique())

# Display the count of each annotated object class
print("\nObject class counts:")
print(annotations["Class_Label"].value_counts())


# Step 3: Load a pre-trained YOLO model
# The model weights are downloaded automatically if needed
model = YOLO("yolo11n.pt")


# Step 4: Open the CCTV video
cap = cv2.VideoCapture(video_path)

# Stop if OpenCV cannot open the video
if not cap.isOpened():
    raise RuntimeError(
        "Could not open the video. Check its path and file format."
    )

# Read video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Use a safe fallback if the video's FPS is unavailable
if fps <= 0:
    fps = 25.0

print("\nVideo width:", width)
print("Video height:", height)
print("Video FPS:", fps)


# Step 5: Create an output video writer
# Save the annotated video as cctv_detected.mp4
output_path = "cctv_detected.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (width, height)
)

# Stop if the output video cannot be created
if not writer.isOpened():
    cap.release()
    raise RuntimeError("Could not create the output video.")


# Step 6: Process the video frame by frame
try:
    while True:

        # Read the next frame from the video
        success, frame = cap.read()

        # End the loop when no more frames are available
        if not success:
            break

        # Run YOLO object detection on the current frame
        # conf sets the minimum confidence threshold
        results = model.predict(
            source=frame,
            conf=0.40,
            verbose=False
        )

        # Draw bounding boxes and labels on the frame
        annotated_frame = results[0].plot()

        # Write the annotated frame to the output video
        writer.write(annotated_frame)

        # Display the detected frame
        cv2.imshow("CCTV Object Detection", annotated_frame)

        # Press Q to stop processing early
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    # Release the input and output video resources
    cap.release()
    writer.release()

    # Close the display window
    cv2.destroyAllWindows()


# Step 7: Display the location of the output video
print("\nObject detection completed.")
print("Detected video saved to:", output_path)
