import pyrealsense2 as rs
import time
import numpy as np

def calibrate_time_offset(pipeline, num_samples=50):
    """
    Collect synchronization points and compute the offset between the camera sensor timestamps (microseconds)
    and the host timestamps (nanoseconds).
    Returns: offset (host nanoseconds = camera nanoseconds + offset_nanoseconds)
    """
    samples = []
    print(f"Collecting {num_samples} synchronization points...")
    
    for _ in range(num_samples):
        host_ts_ns = time.time_ns()
        # Get camera frame and corresponding sensor timestamp (microseconds)
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        
        if not color_frame:
            continue
        
        camera_ts_us = color_frame.get_frame_metadata(rs.frame_metadata_value.sensor_timestamp)
        camera_ts_ns = camera_ts_us * 1000
        
        samples.append((camera_ts_ns, host_ts_ns))
        time.sleep(0.01)  # Wait 10ms between samples
    
    if not samples:
        raise ValueError("No valid synchronization points collected")
    
    # Separate sample data
    camera_timestamps, host_timestamps = zip(*samples)
    
    # Calculate offset: host_time_ns = camera_time_ns + offset_ns
    offsets = np.array(host_timestamps) - np.array(camera_timestamps)
    avg_offset = np.mean(offsets)  
    
    print(f"Time calibration complete, average offset: {avg_offset:.0f} ns ({avg_offset/1e6:.6f} ms)")
    return avg_offset

def camera_ts_to_host_ts(camera_ts_us, offset_ns):
    """
    Convert camera sensor timestamp (microseconds) to host absolute time (nanoseconds, Unix timestamp)
    """
    camera_ts_ns = camera_ts_us * 1000
    return camera_ts_ns + offset_ns

def main():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    
    try:
        # 1. Calibrate time offset
        offset = calibrate_time_offset(pipeline)
        
        # 2. Capture frames in real time and convert timestamps
        print("\nStarting real-time timestamp conversion... (Ctrl+C to exit)")
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            
            camera_ts_us = color_frame.get_frame_metadata(rs.frame_metadata_value.sensor_timestamp)
            host_time_ns = camera_ts_to_host_ts(camera_ts_us, offset)
            
            # Convert to seconds for display (9 decimal places for nanosecond precision)
            host_time_sec = host_time_ns / 1e9
            current_host_time_sec = time.time()
            
            # Print results
            print(f"Camera sensor timestamp: {camera_ts_us} μs -> Converted host time: {host_time_sec:.9f} s")
            print(f"Current host time: {current_host_time_sec:.9f} s, Error: {abs(host_time_sec - current_host_time_sec):.9f} s")
            print("-" * 80)
            time.sleep(1)  # Print once per second
    
    except KeyboardInterrupt:
        print("\nProgram exited")
    finally:
        pipeline.stop()

if __name__ == "__main__":
    main()
