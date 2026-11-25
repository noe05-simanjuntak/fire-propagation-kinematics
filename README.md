# Real-Time Fire Propagation Kinematics Tracker (YOLOv8)

A computer vision and visual metrology pipeline built with **Ultralytics YOLOv8** and **OpenCV** to detect fire regions in video streams, compute a unified center of mass, and track real-time propagation kinematics.

## Sample Output

!\[fire-propagation-kinematics HUD\](assets/Detection_Result.jpg)

## Key Features & Tracked Metrics

* **Unified Bounding Box & Centroid (`cx`, `cy`)**: Aggregates multiple frame-level fire detections into a single outer bounding region to track the overall fire front and center of mass.

* **Projected Area (`px^2`)**: Computes the real-time projected bounding area (`width * height`).

* **Expansion Rate (`px^2/s`)**: Calculates the temporal derivative of the projected area (`dA/dt`) to quantify fire growth dynamics.

* **Smoothed Centroid Velocity (`px/s`) & Direction (`deg`)**: Applies a sliding-window moving average (`WINDOW_SIZE = 15`) on centroid displacement (`dx`, `dy`) to compute propagation speed and directional angle via `atan2`.

* **1-Second Trajectory Projection**: Renders a predicted bounding zone and directional velocity vector overlay on the live dashboard.

* **Telemetry & Data Logging**: Includes an on-screen HUD and built-in CSV export (`Kinematics_Data_*.csv`) for post-experiment analysis.

## Interactive Controls

| Key | Action | 
| ----- | ----- | 
| `q` | Quit the application | 
| `p` | Pause / resume video playback | 
| `s` | Capture and save a timestamped screenshot (`Detection_Result_*.jpg`) | 
| `r` | Start / stop recording real-time kinematics telemetry to CSV | 

## Project Structure

```
fire-propagation-kinematics/
├── assets/
│   └── detection_sample.jpg
├── main.py
├── requirements.txt
├── .gitignore
└── README.md

```

## Setup & Usage

1. Install the required dependencies:

   ```
   pip install -r requirements.txt
   
   ```

2. Place your trained YOLOv8 weights (`best.pt`) in the project root directory and your input video at `dataset_video/experiment_1.mp4`.

3. Run the tracker:

   ```
   python main.py
   
   ```
