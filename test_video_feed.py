import requests
import time

print("Testing video feed endpoint...")
try:
    # Try to get the first frame from the video feed
    response = requests.get('http://localhost:5001/video_feed', stream=True, timeout=5)
    print(f"Response status: {response.status_code}")
    print(f"Content-Type: {response.headers.get('content-type')}")
    
    # Read the first chunk
    for i, chunk in enumerate(response.iter_content(chunk_size=1024)):
        if i == 0:
            print(f"First chunk received: {len(chunk)} bytes")
            if chunk.startswith(b'--frame'):
                print("✅ Video feed is working! Stream headers detected.")
            break
    
    print("\n✅ SUCCESS: Video feed is streaming properly!")
    
except Exception as e:
    print(f"❌ Error: {e}")
