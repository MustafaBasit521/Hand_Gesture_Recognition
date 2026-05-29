"""
virtual_canvas.py — Provides a virtual drawing surface for the Air-Draw application.

The VirtualCanvas class maintains two drawing targets:
  1. The live webcam frame  → gives the user real-time visual feedback of their strokes
  2. An internal black canvas → stores the clean digit drawing for later model inference

Drawing is controlled by finger position coordinates passed in from the hand tracking module.
"""

import cv2     # OpenCV: used for drawing lines on images
import numpy as np  # NumPy: used for creating and zeroing out the canvas array


class VirtualCanvas:
    """
    A virtual whiteboard that draws strokes based on finger tip coordinates.

    Maintains both a user-visible overlay on the camera feed and a separate
    clean black canvas (used as the digit image for CNN prediction).
    """

    def __init__(self, width, height):
        """
        Initialize the canvas with a given size and default drawing settings.

        Args:
            width  (int): Width  of the canvas in pixels (should match webcam frame width).
            height (int): Height of the canvas in pixels (should match webcam frame height).
        """

        # Store canvas dimensions for use in resetting/creating blank arrays
        self.width = width    # Horizontal size of the drawing area
        self.height = height  # Vertical size of the drawing area

        # Create a blank (all-black) 3-channel canvas as a NumPy array
        # Shape: (height, width, 3) → 3 channels = BGR color space (matches OpenCV convention)
        # dtype uint8: pixel values range 0–255 (standard 8-bit image)
        # np.zeros fills everything with 0 → pure black background
        self.canvas = np.zeros((height, width, 3), np.uint8)

        # Store the previous finger tip position so we can draw a continuous line
        # between the previous and current position (rather than just isolated dots)
        # Initialized to (0, 0) which serves as a sentinel meaning "no previous point yet"
        self.xp, self.yp = 0, 0

        # Drawing color in BGR format: (255, 255, 255) = white
        # White on a black canvas mimics the MNIST dataset style (white digit, black background)
        self.draw_color = (255, 255, 255)

        # Thickness of the brush stroke in pixels
        # A thickness of 15 produces a bold stroke that is visible and suitable for digit recognition
        self.brush_thickness = 15

    def draw(self, img, x1, y1, is_drawing):
        """
        Draw a stroke on both the webcam frame and the internal canvas.

        Called every frame while the hand is tracked. When is_drawing is True,
        a line segment is drawn from the previous position to the current position.
        When is_drawing is False, the previous position is reset so the next
        stroke starts fresh without a connecting line.

        Args:
            img        (numpy.ndarray): The live BGR webcam frame to draw visual feedback on.
            x1         (int): Current x-coordinate (horizontal) of the fingertip in pixels.
            y1         (int): Current y-coordinate (vertical)   of the fingertip in pixels.
            is_drawing (bool): True  = draw mode (finger is in drawing position).
                               False = idle mode (stop drawing, reset previous point).

        Returns:
            img (numpy.ndarray): The webcam frame with the stroke drawn on it.
        """

        if is_drawing:
            # ── HANDLE FIRST POINT OF A NEW STROKE ──────────────────────────

            # If previous point is (0, 0) it means this is the start of a new stroke
            # Set the previous point equal to the current point so the first
            # segment has zero length (a dot) rather than a line from the origin
            if self.xp == 0 and self.yp == 0:
                self.xp, self.yp = x1, y1

            # ── DRAW LINE SEGMENT ON BOTH SURFACES ───────────────────────────

            # Draw on the live webcam frame (provides real-time visual feedback to the user)
            # Connects the previous fingertip position to the current one with a colored line
            cv2.line(img, (self.xp, self.yp), (x1, y1), self.draw_color, self.brush_thickness)

            # Draw the same stroke on the internal black canvas (stores the digit for CNN input)
            # This canvas is kept separate so it remains clean without webcam background noise
            cv2.line(self.canvas, (self.xp, self.yp), (x1, y1), self.draw_color, self.brush_thickness)

            # Update previous point to the current point for the next frame's line segment
            self.xp, self.yp = x1, y1

        else:
            # ── RESET PREVIOUS POINT (not drawing) ───────────────────────────

            # When the user lifts their finger or switches to a non-drawing gesture,
            # reset the previous coordinates to (0, 0) so the next time drawing starts
            # we don't accidentally connect it to the old position
            self.xp, self.yp = 0, 0

        return img  # Return the updated webcam frame

    def get_canvas(self):
        """
        Return the internal drawing canvas.

        Used when the application needs to extract the drawn digit image
        and pass it through the preprocessing pipeline and CNN model for prediction.

        Returns:
            canvas (numpy.ndarray): The BGR canvas array with all drawn strokes.
        """
        return self.canvas  # Return the stored black canvas with the drawn digit

    def clear_canvas(self):
        """
        Erase all drawings by resetting the canvas to all black.

        Called when the user makes a "clear" gesture (e.g., open palm).
        Also resets the previous point tracker so the next stroke starts fresh.
        """

        # Overwrite the canvas with a brand-new all-zeros (black) array of the same size
        self.canvas = np.zeros((self.height, self.width, 3), np.uint8)

        # Reset the previous drawing position to the sentinel value (0, 0)
        self.xp, self.yp = 0, 0
