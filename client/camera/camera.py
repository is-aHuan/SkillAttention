import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
import time


class Realsense():
    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        # self.spatial_filter = rs.spatial_filter()
        # self.spatial_filter.set_option(rs.option.holes_fill, 5)

        # Start camera stream
        self.pipeline.start(config)

        device = self.pipeline.get_active_profile().get_device()
        sensor = device.query_sensors()[1]
        sensor.set_option(rs.option.enable_auto_exposure, 1)

        self.s = device.first_roi_sensor()
        self.get_frame(drop=5)  # discard garbage frames and initialize intrinsic parameters

    def get_frame(self, drop=0):
        # discard garbage frames
        for i in range(drop):
            frames = self.pipeline.wait_for_frames()

        frames = self.pipeline.wait_for_frames()

        # align to color frame
        align = rs.align(rs.stream.color)
        frames = align.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            raise Exception("Unable to get frame data")

        self.depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

        # Convert depth data to numpy array
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Dilate to fill holes
        kernel = np.ones((5, 5), np.uint8)
        depth_image = cv2.dilate(depth_image, kernel, iterations=3)

        return color_image, depth_image, depth_frame
    
    def get_frame_timestamp(self):
        frames = self.pipeline.wait_for_frames()
        host_ts_us = frames.get_timestamp() * 1_000_000  # convert to nanoseconds

        # align to color frame
        align = rs.align(rs.stream.color)
        frames = align.process(frames)
        color_frame = frames.get_color_frame()

        color_image = np.asanyarray(color_frame.get_data())

        return color_image, host_ts_us

    def vis_stream(self):
        while True:
            # Wait for a coherent pair of frames: depth and color
            color_image, depth_image, depth_frame = self.get_frame()
            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            # Stack both images horizontally
            images1 = np.hstack((color_image, depth_colormap))
            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images1)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break

    def vis_stream_rotated(self):
        while True:
            # Wait for a coherent pair of frames: depth and color
            color_image, depth_image, depth_frame = self.get_frame()
            color_image_rotated = cv2.rotate(color_image, cv2.ROTATE_90_CLOCKWISE)
            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_image_rotated = cv2.rotate(depth_image, cv2.ROTATE_90_CLOCKWISE)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image_rotated, alpha=0.03), cv2.COLORMAP_JET)
            # Stack both images horizontally
            images1 = np.hstack((color_image_rotated, depth_colormap))
            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images1)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break

    def get_coordinate_from_pic(self, x, y, depth_frame):
        """
        Get camera coordinates from image pixel coordinates
        """
        depth = depth_frame.get_distance(x, y)
        if depth == 0:
            raise Exception("Failed to get depth")

        # Get camera intrinsics
        self.depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

        # Deproject pixel and depth to camera coordinates
        camera_coordinate = rs.rs2_deproject_pixel_to_point(self.depth_intrinsics, (x, y), depth)
        print(f"Pixel ({x}, {y}) in camera coordinates: {camera_coordinate}, object center depth: {depth}")

        return camera_coordinate

    def get_point_cloud(self, depth_image, mask=None, vis_flag=1):

        # Convert depth units to meters
        depth_image = depth_image / 1000
        rows, cols = depth_image.shape

        # Use meshgrid to generate row and column grid coordinates
        rows_grid, cols_grid = np.meshgrid(np.arange(rows), np.arange(cols))
        # Create grid coordinates for rows and columns
        pc = np.concatenate(
            [rows_grid.T[:, :, np.newaxis], cols_grid.T[:, :, np.newaxis], depth_image[:, :, np.newaxis]], axis=-1) # (H, W, 3)
        # Now pc is a 3D array with last dim length 3, containing [row, col, depth] for each pixel

        pc = pc.reshape(-1, 3) # (H*W, 3)
        # Compute coordinates in the camera coordinate system
        fx, fy, cx, cy = self.depth_intrinsics.fx, self.depth_intrinsics.fy, self.depth_intrinsics.ppx, self.depth_intrinsics.ppy
        pc[:, 0] = (pc[:, 0] - cy) / fy * pc[:, 2]
        pc[:, 1] = (pc[:, 1] - cx) / fx * pc[:, 2]


        # Display the full image point cloud
        if vis_flag in [2]:
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(pc)  # convert points to Open3D Vector3dVector
            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
            o3d.visualization.draw_geometries([point_cloud, mesh_frame])

            if mask:
                # If mask provided, set depth outside region to 0 and display point cloud again
                masked_pc = pc.reshape(rows, cols, 3)
                masked_pc[mask != True, 2] = 0
                masked_pc = masked_pc.reshape(-1, 3)
                point_cloud = o3d.geometry.PointCloud()
                point_cloud.points = o3d.utility.Vector3dVector(masked_pc)
                o3d.visualization.draw_geometries([point_cloud])

        pc = pc.reshape(rows, cols, 3)
        pc = pc[:, :, [1, 0, 2]]  # Switch to row-column ordering

        return pc

    def get_flat_point_cloud_optmized_by_mask(self, depth_image, mask, vis_flag=1):
        # use mask to complete depth hole of flat area
        # Convert depth units to meters
        depth_image = depth_image / 1000
        rows, cols = depth_image.shape
        # use mask to complete depth hole
        mask = mask.astype(bool)
        mask_depth = depth_image * mask

        # ============old===============
        # mask_depth[mask_depth > 0.72] = 0
        # avg_mask_depth = np.mean(mask_depth[mask_depth > 0.1])
        # ============old===============
        # ============new===============

        # Get point cloud
        depth_image = depth_image * 1000
        pc = self.get_point_cloud(depth_image, vis_flag=vis_flag) # (H, W, 3)
        # Filter point cloud by mask
        pc = pc.reshape(-1, 3)  # array (H*W, 3)
        H, W = mask.shape
        mask = mask.flatten()  # array (H*W) dtype=bool

        # Remove ground points from mask
        index = pc[:, 2] > 0.72
        mask[index] = False

        pc = pc[mask == True] # (n, 3)
        # Filter point cloud by depth thresholds
        pc = pc[pc[:, 2] > 0.1]
        pc = pc[pc[:, 2] < 0.72]
        # Plane detection
        pc_o3d = o3d.geometry.PointCloud()  # create point cloud object
        pc_o3d.points = o3d.utility.Vector3dVector(pc)
        plane_model, inliers = pc_o3d.segment_plane(distance_threshold=0.005,  # max distance from point to plane to be considered an inlier
                                                ransac_n=5,  # number of points to sample for plane estimation
                                                num_iterations=1000)  # number of RANSAC iterations
        # The function performs plane detection and returns the plane equation (ax+by+cz+d=0) and indices of inliers.
        pc_o3d = pc_o3d.select_by_index(inliers)  # Select the plane point cloud by inliers

        pc = np.asarray(pc_o3d.points)
        avg_mask_depth = np.mean(pc[:, 2])
        # ============new===============

        depth_image[depth_image < 0.1] = avg_mask_depth
        depth_image[depth_image > 0.72] = avg_mask_depth
        print(f'avg_mask_depth {avg_mask_depth}')


        # Use meshgrid to generate row and column grid coordinates
        rows_grid, cols_grid = np.meshgrid(np.arange(rows), np.arange(cols))
        # Create grid coordinates for rows and columns
        pc = np.concatenate(
            [rows_grid.T[:, :, np.newaxis], cols_grid.T[:, :, np.newaxis], depth_image[:, :, np.newaxis]], axis=-1) # (H, W, 3)
        # Now pc is a 3D array with last dim length 3, containing [row, col, depth] for each pixel

        pc = pc.reshape(-1, 3) # (H*W, 3)
        # Compute coordinates in the camera coordinate system
        fx, fy, cx, cy = self.depth_intrinsics.fx, self.depth_intrinsics.fy, self.depth_intrinsics.ppx, self.depth_intrinsics.ppy
        pc[:, 0] = (pc[:, 0] - cy) / fy * pc[:, 2]
        pc[:, 1] = (pc[:, 1] - cx) / fx * pc[:, 2]

        # Display the full image point cloud
        if vis_flag in [2]:
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(pc)  # convert points to Open3D Vector3dVector
            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
            o3d.visualization.draw_geometries([point_cloud, mesh_frame])

            # if mask:
            #     # If mask provided, set depth outside region to 0 and display point cloud again
            #     masked_pc = pc.reshape(rows, cols, 3)
            #     masked_pc[mask != True, 2] = 0
            #     masked_pc = masked_pc.reshape(-1, 3)
            #     point_cloud = o3d.geometry.PointCloud()
            #     point_cloud.points = o3d.utility.Vector3dVector(masked_pc)
            #     o3d.visualization.draw_geometries([point_cloud])

        pc = pc.reshape(rows, cols, 3)
        pc = pc[:, :, [1, 0, 2]]  # Switch to row-column ordering

        return pc, mask.reshape(H, W)

    def set_roi(self, x1=20, y1=20, x2=580, y2=420, vis_flag=1):

        roi = self.s.get_region_of_interest()
        print(f'before roi {roi.min_x}, {roi.min_y}, {roi.max_x}, {roi.max_y}')

        if vis_flag in [2]:
            color_image, depth_image, depth_frame = self.get_frame()
            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            cv2.rectangle(color_image, (roi.min_x, roi.min_y), (roi.max_x, roi.max_y), (0, 0, 255), 2)  # Red rectangle, line width 2

            # Stack both images horizontally
            images1 = np.hstack((color_image, depth_colormap))
            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('before setting roi', images1)

        roi.min_x, roi.min_y, roi.max_x, roi.max_y = x1, y1, x2, y2 # column indices
        # Set ROI for metering
        self.s.set_region_of_interest(roi)
        time.sleep(0.2)
        roi = self.s.get_region_of_interest()
        print(f'after roi {roi.min_x}, {roi.min_y}, {roi.max_x}, {roi.max_y}')

        if vis_flag in [2]:
            color_image, depth_image, depth_frame = self.get_frame()
            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            cv2.rectangle(color_image, (roi.min_x, roi.min_y), (roi.max_x, roi.max_y), (0, 0, 255), 2)  # Red rectangle, line width 2

            # Stack both images horizontally
            images1 = np.hstack((color_image, depth_colormap))
            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('after setting roi', images1)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def get_timestamp(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise Exception("Unable to get color frame data")
        # Get camera sensor timestamp (microseconds, relative)
        camera_ts_ns = color_frame.get_timestamp() * 1_000_000  # convert to nanoseconds
        return camera_ts_ns

if __name__ == '__main__':
    cam = Realsense()
    while True:
        cam.vis_stream()

        time.sleep(0.1)