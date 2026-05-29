"""
hand_tracking.py — Real-time hand detection and landmark tracking using MediaPipe.

Provides a HandDetector class that:
  - Detects hands in a video frame and draws landmarks
  - Returns pixel-space coordinates for all 21 hand landmarks
  - Determines which fingers are currently raised (used for gestures)
"""

import cv2        # OpenCV: library for image processing and drawing
import mediapipe as mp  # MediaPipe: Google's ML framework for hand detection


class HandDetector:
    """
    Wraps MediaPipe Hands to provide easy hand detection, landmark extraction,
    and finger-state detection for gesture-based applications.
    """

    def __init__(self, mode=False, max_hands=2, detection_con=0.5, track_con=0.5):
        """
        Initialize the HandDetector and set up the MediaPipe Hands model.

        Args:
            mode (bool): False = video stream (tracking mode, faster).
                         True  = static images (full detection every frame).
            max_hands (int): Maximum number of hands to detect at once.
            detection_con (float): Confidence threshold for initial hand detection (0–1).
            track_con (float): Confidence threshold for tracking across frames (0–1).
        """

        # Store config values so they can be passed to the MediaPipe model
        self.mode = mode                   # Static or video mode
        self.max_hands = max_hands         # Max simultaneous hands to detect
        self.detection_con = detection_con # Min detection confidence
        self.track_con = track_con         # Min tracking confidence

        # Access the MediaPipe Hands module (contains Hands class and landmark IDs)
        self.mp_hands = mp.solutions.hands

        # Instantiate the MediaPipe Hands model with our configuration
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,                  # Stream vs static mode
            max_num_hands=self.max_hands,                 # Detect at most this many hands
            min_detection_confidence=self.detection_con,  # Threshold for new detections
            min_tracking_confidence=self.track_con,       # Threshold for ongoing tracking
        )

        # Utility for drawing landmarks and bone-like connections on images
        self.mp_draw = mp.solutions.drawing_utils

        # IDs of the 5 fingertip landmarks (in order: thumb, index, middle, ring, pinky)
        # Full landmark map: 0=Wrist, 4=Thumb tip, 8=Index tip, 12=Middle tip, 16=Ring tip, 20=Pinky tip
        self.tip_ids = [4, 8, 12, 16, 20]

    def find_hands(self, img, draw=True):
        """
        Detect hands in a BGR image frame and optionally draw the landmarks.

        Args:
            img (numpy.ndarray): BGR image (from OpenCV/webcam).
            draw (bool): If True, draw landmarks and connections on img.

        Returns:
            img (numpy.ndarray): The image, optionally with landmarks drawn on it.
        """

        # Convert BGR → RGB because MediaPipe requires RGB input
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Run MediaPipe hand detection; results contain landmark data for all detected hands
        self.results = self.hands.process(img_rgb)

        # If one or more hands were detected in this frame
        if self.results.multi_hand_landmarks:

            # Loop over each detected hand's landmark set
            for hand_lms in self.results.multi_hand_landmarks:
                if draw:
                    # Draw all 21 landmark dots and the connecting bone lines on the original BGR image
                    self.mp_draw.draw_landmarks(img, hand_lms, self.mp_hands.HAND_CONNECTIONS)

        return img  # Return the annotated (or original) image

    def find_position(self, img, hand_no=0, draw=True):
        """
        Get pixel coordinates for all 21 landmarks of a specified hand.

        MediaPipe returns normalized coordinates (0.0–1.0). This method converts
        them to actual pixel values using the image dimensions.

        Args:
            img (numpy.ndarray): Current BGR frame (used for width/height).
            hand_no (int): Which hand to read (0 = first detected hand).
            draw (bool): If True, draw a circle at each landmark.

        Returns:
            lm_list (list): List of [id, x_px, y_px] for each of the 21 landmarks.
        """

        # Clear the landmark list before filling it for this frame
        self.lm_list = []

        # Only proceed if at least one hand was found
        if self.results.multi_hand_landmarks:

            # Pick the hand at the given index (0 = first hand found)
            my_hand = self.results.multi_hand_landmarks[hand_no]

            # Iterate over all 21 landmarks of the chosen hand
            for id, lm in enumerate(my_hand.landmark):

                # Get image dimensions: h = height, w = width, c = channels
                h, w, c = img.shape

                # Convert normalized (0–1) coordinates to pixel coordinates
                # lm.x × width  → column (horizontal pixel position)
                # lm.y × height → row    (vertical pixel position)
                cx, cy = int(lm.x * w), int(lm.y * h)

                # Store [landmark_id, x_pixel, y_pixel] in the list
                self.lm_list.append([id, cx, cy])

                if draw:
                    # Draw a small magenta filled circle at this landmark's pixel position
                    # (255, 0, 255) = magenta in BGR | radius=5 | cv2.FILLED = solid fill
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        return self.lm_list  # Return list of [id, x, y] for all 21 landmarks

    def fingers_up(self):
        """
        Detect which of the 5 fingers are currently extended/raised.

        Logic:
          - Thumb : compared on the X-axis (opens sideways → tip.x < joint.x)
          - Others: compared on the Y-axis (open upward → tip.y < joint.y)
            Note: in OpenCV Y increases downward, so smaller Y = higher on screen.

        Returns:
            fingers (list): [thumb, index, middle, ring, pinky] — 1=up, 0=down.
        """

        fingers = []  # Will hold one 0 or 1 per finger, in order thumb→pinky

        # ── THUMB ─────────────────────────────────────────────────────────────

        # Thumb tip = tip_ids[0] = landmark 4
        # Joint below tip = landmark 3 (tip_ids[0] - 1)
        # Thumb opens horizontally: if tip X < joint X → thumb is extended
        # lm_list[id][1] = x-pixel of that landmark
        if self.lm_list[self.tip_ids[0]][1] < self.lm_list[self.tip_ids[0] - 1][1]:
            fingers.append(1)  # Thumb is up
        else:
            fingers.append(0)  # Thumb is curled

        # ── INDEX, MIDDLE, RING, PINKY ────────────────────────────────────────

        for id in range(1, 5):  # Loop through the 4 non-thumb fingers

            # tip_ids[id]     = landmark ID of the fingertip
            # tip_ids[id] - 2 = landmark ID of the PIP joint (2 joints below the tip)
            # lm_list[...][2] = y-pixel of that landmark
            # If tip Y < PIP joint Y → tip is higher on screen → finger is extended
            if self.lm_list[self.tip_ids[id]][2] < self.lm_list[self.tip_ids[id] - 2][2]:
                fingers.append(1)  # Finger is up
            else:
                fingers.append(0)  # Finger is curled

        return fingers  # [thumb, index, middle, ring, pinky] — 1=up, 0=down
