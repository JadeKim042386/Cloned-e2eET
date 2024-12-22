# e2eET Skeleton Based HGR Using Data-Level Fusion
# pyright: reportGeneralTypeIssues=false
# pyright: reportOptionalMemberAccess=false
# pyright: reportWildcardImportFromLibrary=false
# -----------------------------------------------
"""
Hand Tracking Module
By: Computer Vision Zone
Website: https://www.computervision.zone/
"""

import cv2 as cv
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class HandDetector:
    """
    Finds Hands using the mediapipe library. Exports the landmarks
    in pixel format. Adds extra functionalities like finding how
    many fingers are up or the distance between two fingers. Also
    provides bounding box info of the hand found.
    """

    def __init__(self, mode=False, maxHands=1, detectionCon=0.5, minTrackCon=0.5):
        """
        :param mode: In static mode, detection is done on each image: slower
        :param maxHands: Maximum number of hands to detect
        :param detectionCon: Minimum Detection Confidence Threshold
        :param minTrackCon: Minimum Tracking Confidence Threshold
        """
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.minTrackCon = minTrackCon

        base_options = python.BaseOptions(model_asset_path="./.sources/hand_landmarker.task")
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=maxHands,
            min_tracking_confidence=minTrackCon,
            min_hand_detection_confidence=detectionCon,
        )
        self.hands = vision.HandLandmarker.create_from_options(options)
        self.mpDraw = mp.solutions.drawing_utils
        self.mpDrawStyles = mp.solutions.drawing_styles
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        """
        Finds hands in a BGR image.
        :param img: Image to find the hands in.
        :return: Image with or without drawings
        """
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        imgRGB = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
        self.results = self.hands.detect(imgRGB)
        allHands = []
        h, w, c = img.shape

        if self.results.hand_landmarks:
            for handLms in self.results.hand_landmarks:
                _hand = {}

                ## lmCoords
                lmCoords_2D = []
                lmCoords_3D = []
                xList = []
                yList = []
                for id, lm in enumerate(handLms):
                    lmCoords_3D.append([lm.x, lm.y, lm.z])
                    px, py = int(lm.x * w), int(lm.y * h)
                    lmCoords_2D.append([px, py, 0.0])
                    xList.append(px)
                    yList.append(py)

                ## bbox
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                boxW, boxH = xmax - xmin, ymax - ymin
                bbox = xmin, ymin, boxW, boxH
                cx, cy = bbox[0] + (bbox[2] // 2), bbox[1] + (bbox[3] // 2)

                _hand["lmCoords_2D"] = np.array(lmCoords_2D)
                _hand["lmCoords_3D"] = np.array(lmCoords_3D).round(4)
                _hand["bbox"] = np.array(bbox)
                _hand["center"] = np.array([cx, cy])

                allHands.append(_hand)

                if draw:
                    self.mpDraw.draw_landmarks(
                        img,
                        handLms,
                        self.mpHands.HAND_CONNECTIONS,
                        self.mpDrawStyles.get_default_hand_landmarks_style(),
                        self.mpDrawStyles.get_default_hand_connections_style(),
                    )

        return allHands, img


def main():
    cap = cv.VideoCapture(0)
    detector = HandDetector(detectionCon=0.8, maxHands=2)
    while True:
        # Get image frame
        success, img = cap.read()
        # Find the hand and its landmarks
        hands, img = detector.findHands(img)  # with draw
        # hands = detector.findHands(img, draw=False)  # without draw

        if hands:
            # Hand 1
            hand1 = hands[0]
            lmCoords_2D_1 = hand1["lmCoords_2D"]  # List of 21 Landmark points
            lmCoords_3D_1 = hand1["lmCoords_3D"]  # List of 21 raw x,y,z lms
            bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h
            centerPoint1 = hand1["center"]  # center of the hand cx,cy

            print(np.array(lmCoords_3D_1).round(4).tolist(), end="\n\n")

            if len(hands) == 2:
                # Hand 2
                hand2 = hands[1]
                lmCoords_2D_2 = hand2["lmCoords_2D"]  # List of 21 Landmark points
                lmCoords_3D_2 = hand2["lmCoords_3D"]  # List of 21 raw x,y,z lms
                bbox2 = hand2["bbox"]  # Bounding box info x,y,w,h
                centerPoint2 = hand2["center"]  # center of the hand cx,cy

        # Display
        cv.imshow("Image", cv.flip(img, 1))
        if cv.waitKey(1) & 0xFF == ord(" "):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
