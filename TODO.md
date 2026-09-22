# 3D Printing First Layer Detection Integration TODO

## Completed Steps
- [x] Created detailed edit plan for real ML integration
- [x] User approved plan to update first_layer_detector.html
- [x] Updated first_layer_detector_updated.html: Real API poll (:8080/api/predict), camera feed, thresholds (90+/excellent, 80+/acceptable, <80 warning, <70 emergency), real chart/logs, no fakes

## Remaining Steps (from approved plan)
1. Update first_layer_detector.html: Replace fake JS with real API polling (/api/predict), camera feed (:8080/video_feed), thresholds, chart, logs.
2. Test backend: cd dashboard && ./run.sh (port 8080)
3. Verify live feed, detections, emergency alerts (<70%), warnings (<80%)
4. Debug camera if "NO FEED" (test_camera.py, macOS permissions)
5. Attempt completion with demo command

Progress: 1/5 ✅ Next: Edit HTML file.

