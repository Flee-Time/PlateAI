from ultralytics import YOLO
from collections import Counter
import cv2
import numpy as np
import sys
import easyocr
import os
import re

arg_info = "Usage: main.py (Video Device Number) (Plate Detection Model Location) (Optional commands)\n\
--------------------------Optional Commands----------------------------\n\
    -b    |    Black and white image processing mode\n\
    -v    |    Needed for video file input\n\
    -u    |    Upscales the detected plates for better detection\n\
  TIP: you can combine flags like -bv or -vb etc.\n\
-----------------------------------------------------------------------"


class PlateAi:
    def __init__(self):
        bw_mode = False
        vid_mode = False
        upscale_mode = False

        if len(sys.argv) < 2:
            self.PrintErrorAndExit("No video device or stream url provided.", arg_info)
        elif len(sys.argv) < 3:
            self.PrintErrorAndExit("No plate detection model location provided.", arg_info)
        elif len(sys.argv) == 4:
            if sys.argv[3].startswith("-"):
                for char in sys.argv[3][1:]:
                    if char == "b":
                        bw_mode = True
                    elif char == "v":
                        vid_mode = True
                    elif char == "u":
                        upscale_mode = True
                    else:
                        self.PrintErrorAndExit(f"Unrecognized command '{char}'.", arg_info)
            else:
                self.PrintErrorAndExit(
                    "Invalid argument format. Expected '-' followed by options.",
                    arg_info,
                )

        try:
            stream_url = int(sys.argv[1])
        except ValueError:
            stream_url = str(sys.argv[1])

        # Editable variables
        self.filter_length = 20

        # Class variable setting
        self.stream_url = stream_url
        self.plate_model_location = str(sys.argv[2])
        self.bw_mode = bw_mode
        self.vid_mode = vid_mode
        self.upscale_mode = upscale_mode

        # Regular expression pattern for Turkish license plates
        self.turkey_plate_pattern = r"^\d{2}[A-Z]{1,3}\d{2,4}$"

        # Model loading at init
        if os.path.exists(self.plate_model_location):
            try:
                self.plate_model = YOLO(self.plate_model_location)
            except Exception as e:
                self.PrintErrorAndExit(
                    f"Error loading YOLO model from {self.plate_model_location}: {e}",
                    "",
                )
        else:
            self.PrintErrorAndExit(f"Error: Model file not found at {self.plate_model_location}", "")

        try:
            self.reader = easyocr.Reader(["en"])
        except Exception as e:
            self.PrintErrorAndExit(f"Error initializing EasyOCR reader: {e}", "")

    def PrintErrorAndExit(self, message, arg_error_info):
        print(f"Error: {message}")
        if arg_error_info != "":
            print(arg_error_info)
        exit(1)

    def Exit(self, state="Quit"):
        if state == "KeyboardInterrupt":
            print("KeyboardInterrupt detected, exiting.")
        else:
            print("Exiting.")
        self.capture.release()
        cv2.destroyAllWindows()

    def InitCapture(self):
        if self.vid_mode:
            self.capture = cv2.VideoCapture(self.stream_url)
        else:
            self.capture = cv2.VideoCapture(self.stream_url, cv2.CAP_DSHOW)
        if not self.capture.isOpened():
            print("Error: Could not open video file.")
            exit(1)

    def IsValidPlate(self, plate):
        if re.match(self.turkey_plate_pattern, plate):
            return True
        else:
            return False

    def RawCamLoop(self):
        try:
            plates = []
            cnt = 0
            corrected_plate = None
            last_plate = None

            while True:
                ret, frame = self.capture.read()
                if not ret:
                    break

                frame, cropped = self.ProcessRawCam(frame)

                if cropped is not None:
                    result = self.reader.readtext(cropped)

                    plate = "".join(text.strip().upper().replace(" ", "") for (bbox, text, prob) in result)

                    if plate.startswith("TR"):
                        plate.replace("TR", "")

                    if self.IsValidPlate(plate):
                        plates.append(plate)
                        cnt += 1

                    if cnt >= self.filter_length:
                        pt = Counter(plates)
                        corrected_plate = pt.most_common(1)[0][0]
                        # Update last_plate only if the new corrected_plate is valid
                        if corrected_plate and last_plate != corrected_plate:
                            last_plate = corrected_plate
                            print(corrected_plate)
                            # Reset counters and clear plates list
                        cnt = 0
                        plates.clear()
                    else:
                        corrected_plate = None

                # Display the frame
                cv2.imshow("Object Detection", frame)
                if cv2.waitKey(1) == ord("q"):
                    break
            self.Exit()
        except KeyboardInterrupt:
            self.Exit("KeyboardInterrupt")

    def ProcessRawCam(self, frame):
        # Perform object detection
        results = self.plate_model(frame, stream=True, verbose=False)
        cropped = None
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Plate only masking, doesnt suport multiple detections rn
                mask = np.zeros(frame.shape[:2], np.uint8)
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                masked = cv2.bitwise_and(frame, frame, mask=mask)
                (x, y) = np.where(mask == 255)
                (topx, topy) = (np.min(x), np.min(y))
                (bottomx, bottomy) = (np.max(x), np.max(y))
                cropped = masked[topx : bottomx + 1, topy : bottomy + 1]

                if self.bw_mode:
                    cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                    cropped_gray = cv2.bilateralFilter(cropped_gray, 13, 30, 30)
                    (thresh, cropped) = cv2.threshold(cropped_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

                if self.upscale_mode:
                    cropped = cv2.resize(cropped, (512, 128))

                cv2.imshow("Rectangular Mask", cropped)

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # Display confidence and class name
                confidence = np.round(box.cpu().conf[0], decimals=2)
                text = f"Plate: {confidence:.2f}"
                org = (x1, y1 - 10)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                color = (0, 255, 0)
                thickness = 2
                cv2.putText(frame, text, org, font, font_scale, color, thickness)
        return frame, cropped


program = PlateAi()
program.InitCapture()
program.RawCamLoop()
