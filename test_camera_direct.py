import cv2
import time

print("Testing camera...")

# Try camera 0
print("\n=== Testing Camera 0 ===")
cap0 = cv2.VideoCapture(0)
print(f"Camera 0 opened: {cap0.isOpened()}")

if cap0.isOpened():
    cap0.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    for i in range(5):
        ret, frame = cap0.read()
        print(f"Frame {i}: ret={ret}, shape={frame.shape if frame is not None else None}, size={frame.size if frame is not None else None}")
        if ret and frame is not None and frame.size > 0:
            print(f"Camera 0 is working! Got valid frame")
            break
    cap0.release()
else:
    print("Camera 0 failed to open")

# Try camera 1
print("\n=== Testing Camera 1 ===")
cap1 = cv2.VideoCapture(1)
print(f"Camera 1 opened: {cap1.isOpened()}")

if cap1.isOpened():
    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    for i in range(5):
        ret, frame = cap1.read()
        print(f"Frame {i}: ret={ret}, shape={frame.shape if frame is not None else None}, size={frame.size if frame is not None else None}")
        if ret and frame is not None and frame.size > 0:
            print(f"Camera 1 is working! Got valid frame")
            break
    cap1.release()
else:
    print("Camera 1 failed to open")

print("\n=== Test Complete ===")
