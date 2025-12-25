"""
Motion Detector for Automatic Chess Detection
==============================================
Detects motion in board area to automatically pause/resume detection.
Replaces manual button trigger with intelligent motion sensing.

Author: Chess Detection Improved Team
Date: December 2025
"""

import cv2
import numpy as np
import time
from collections import deque


class MotionDetector:
    """
    Intelligent motion detector for automatic detection control.
    
    Features:
    - Motion detection in board ROI only
    - Temporal smoothing to avoid flickering
    - Configurable sensitivity
    - Debouncing for stable state transitions
    """
    
    def __init__(self, 
                 motion_threshold=1500,
                 history_size=5,
                 stable_frames_required=3,
                 min_area=500):
        """
        Initialize motion detector.
        
        Args:
            motion_threshold: Minimum pixel change to consider as motion
            history_size: Number of frames to keep for smoothing
            stable_frames_required: Frames needed for state change
            min_area: Minimum contour area to consider
        """
        self.motion_threshold = motion_threshold
        self.history_size = history_size
        self.stable_frames_required = stable_frames_required
        self.min_area = min_area
        
        # State tracking
        self.previous_frame = None
        self.motion_history = deque(maxlen=history_size)
        self.is_board_clear = True
        self.stable_frame_count = 0
        
        # Performance tracking
        self.last_motion_time = 0
        self.detection_paused_time = 0
        
    def detect_motion(self, frame, board_roi=None):
        """
        Detect motion in frame (or specific ROI).
        
        Args:
            frame: Current frame (BGR)
            board_roi: Optional ROI coordinates (x, y, w, h)
        
        Returns:
            bool: True if motion detected, False otherwise
        """
        # Extract ROI if specified
        if board_roi is not None:
            x, y, w, h = board_roi
            roi = frame[y:y+h, x:x+w]
        else:
            roi = frame
        
        # Convert to grayscale and blur
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # First frame initialization
        if self.previous_frame is None:
            self.previous_frame = gray
            return False
        
        # Compute frame difference
        frame_delta = cv2.absdiff(self.previous_frame, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        
        # Dilate threshold image to fill holes
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(
            thresh.copy(), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Calculate total motion
        motion_pixels = 0
        for contour in contours:
            if cv2.contourArea(contour) < self.min_area:
                continue
            motion_pixels += cv2.contourArea(contour)
        
        # Update previous frame
        self.previous_frame = gray
        
        # Determine if motion is significant
        has_motion = motion_pixels > self.motion_threshold
        
        # Add to history for smoothing
        self.motion_history.append(has_motion)
        
        # Update last motion time
        if has_motion:
            self.last_motion_time = time.time()
        
        return has_motion
    
    def is_stable(self, desired_state):
        """
        Check if state has been stable for required frames.
        
        Args:
            desired_state: True for clear, False for motion
        
        Returns:
            bool: True if stable enough for state transition
        """
        if len(self.motion_history) < self.history_size:
            return False
        
        # Count frames matching desired state
        if desired_state:  # Want board clear (no motion)
            matching_frames = sum(1 for m in self.motion_history if not m)
        else:  # Want motion detected
            matching_frames = sum(1 for m in self.motion_history if m)
        
        return matching_frames >= self.stable_frames_required
    
    def should_detect(self, frame, board_roi=None):
        """
        Main decision function: should detection run?
        
        Args:
            frame: Current frame
            board_roi: Optional board ROI
        
        Returns:
            bool: True if detection should run, False if should pause
        """
        # Detect motion in current frame
        has_motion = self.detect_motion(frame, board_roi)
        
        # State machine logic
        if self.is_board_clear:
            # Currently in DETECTING state
            # Check if motion appeared (hand entered board)
            if has_motion and self.is_stable(False):
                # Transition to PAUSED state
                self.is_board_clear = False
                self.stable_frame_count = 0
                self.detection_paused_time = time.time()
                print("🚫 Motion detected - Detection PAUSED")
                return False
            return True
        else:
            # Currently in PAUSED state
            # Check if motion stopped (hand left board)
            if not has_motion and self.is_stable(True):
                # Transition to DETECTING state
                self.is_board_clear = True
                self.stable_frame_count = 0
                pause_duration = time.time() - self.detection_paused_time
                print(f"✅ Board clear - Detection RESUMED (paused {pause_duration:.1f}s)")
                return True
            return False
    
    def get_motion_percentage(self):
        """
        Get percentage of recent frames with motion.
        
        Returns:
            float: Percentage (0-100)
        """
        if len(self.motion_history) == 0:
            return 0.0
        return (sum(self.motion_history) / len(self.motion_history)) * 100
    
    def reset(self):
        """Reset detector state."""
        self.previous_frame = None
        self.motion_history.clear()
        self.is_board_clear = True
        self.stable_frame_count = 0
    
    def get_status(self):
        """
        Get current detector status.
        
        Returns:
            dict: Status information
        """
        return {
            'board_clear': self.is_board_clear,
            'motion_percentage': self.get_motion_percentage(),
            'time_since_motion': time.time() - self.last_motion_time if self.last_motion_time > 0 else 0,
            'should_detect': self.is_board_clear
        }
