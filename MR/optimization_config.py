# Configuration for Hyperparameter Optimization
# Defines the search space for the OC-SORT + ByteTrack + Interpolation tracker

SEARCH_SPACE = {
    # --- Detection Thresholds ---
    # High confidence threshold for track initialization (ByteTrack high bin)
    # Range: 0.3 to 0.7 covers the typical "high confidence" spectrum.
    "CONFIDENCE_THRESHOLD": {"type": "float", "low": 0.3, "high": 0.7},
    
    # Low confidence threshold for second association stage (ByteTrack low bin)
    # Must be lower than CONFIDENCE_THRESHOLD.
    "CONFIDENCE_LOW": {"type": "float", "low": 0.01, "high": 0.2},
    
    # --- Association ---
    # IoU threshold for matching detections to tracks.
    "IOU_THRESHOLD": {"type": "float", "low": 0.1, "high": 0.5},
    
    # --- Track Lifecycle ---
    # Maximum number of frames to keep a lost track alive.
    "MAX_AGE": {"type": "int", "low": 15, "high": 120},
    
    # Minimum number of consecutive hits to confirm a track.
    "MIN_HITS": {"type": "int", "low": 1, "high": 5},
    
    # --- OC-SORT Specific ---
    # Inertia weight for the cost matrix (smoothness vs position).
    "INERTIA": {"type": "float", "low": 0.1, "high": 0.5},
    
    # Time step difference for velocity calculation.
    "DELTA_T": {"type": "int", "low": 1, "high": 5},
    
    # --- Post-Processing ---
    # Maximum gap size (in frames) to fill via linear interpolation.
    "MAX_GAP": {"type": "int", "low": 5, "high": 60}
}
