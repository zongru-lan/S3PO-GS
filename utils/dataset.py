import csv
import glob
import os

import cv2
import numpy as np
import torch
import trimesh
from PIL import Image
import json
from pathlib import Path

from gaussian_splatting.utils.graphics_utils import focal2fov
from scipy.spatial.transform import Rotation as R

try:
    import pyrealsense2 as rs
except Exception:
    pass

# We retain the input interfaces for ground-truth depth and other monocular depth estimations (e.g., from DepthAnything).
# In RGB-only scenarios, the first channel of the RGB image is used as a placeholder for depth input.

## ====================================data parser========================================
class dl3dvParser:
    def __init__(self, input_folder, config):
        self.input_folder = input_folder
        self.begin = config["Dataset"]["begin"]
        self.end = config["Dataset"]["end"]
        
        self.color_paths = sorted(glob.glob(f"{self.input_folder}/rgb/*.png"))[self.begin:self.end]
        self.depth_paths = sorted(glob.glob(f"{self.input_folder}/rgb/*.png"))[self.begin:self.end]
        self.mono_depth_paths = sorted(glob.glob(f"{self.input_folder}/rgb/*.png"))[self.begin:self.end]
        self.n_img = len(self.color_paths)
        
        self.load_poses(os.path.join(self.input_folder, "cameras.json"))

    def load_poses(self, pose_file):
        """ Read camera poses from camera.json and convert them to 4×4 matrices """
        self.poses = []
        self.frames = []

        with open(pose_file, "r") as f:
            all_poses = json.load(f)

        selected_poses = all_poses[self.begin:self.end]
        init_trans = np.array(selected_poses[0]["cam_trans"])

        for i, pose in enumerate(selected_poses):
            qx, qy, qz, qw = pose["cam_quat"]
            tx, ty, tz = pose["cam_trans"]

            rotation_matrix = R.from_quat([qx, qy, qz, qw]).as_matrix()
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = rotation_matrix
            transform_matrix[:3, 3] = [tx, ty, tz] - init_trans 
    
            inv_pose = np.linalg.inv(transform_matrix)
            self.poses.append(inv_pose)  
            frame = {
                "file_path": self.color_paths[i],
                "depth_path": self.color_paths[i],
                "mono_depth_path": self.color_paths[i],
                "transform_matrix": transform_matrix.tolist(),  
            }
            self.frames.append(frame)

class RailwayParser:
    def __init__(self, input_folder, config):
        import pandas as pd

        self.input_folder = input_folder
        dataset_cfg = config["Dataset"]
        self.begin = dataset_cfg.get("begin", 0) or 0
        self.end = dataset_cfg.get("end")
        self.gt_pose_tolerance_sec = float(dataset_cfg.get("gt_pose_tolerance_sec", 0.05))
        self.scene = dataset_cfg.get("scene") or os.path.basename(os.path.normpath(input_folder))
        self.gt_pose_root = dataset_cfg.get(
            "gt_pose_root",
            os.path.join(dataset_cfg.get("dataset_root", os.path.dirname(input_folder)), "gt_poses"),
        )

        self.color_paths = self._load_color_paths(input_folder)
        self.color_paths = self.color_paths[self.begin:self.end]
        self.depth_paths = self.color_paths
        self.mono_depth_paths = self.color_paths
        self.n_img = len(self.color_paths)
        if self.n_img == 0:
            raise FileNotFoundError(f"No railway images found in {input_folder}")

        self.frame_ids = [self._frame_id_from_path(path) for path in self.color_paths]
        self.image_names = [os.path.basename(path) for path in self.color_paths]
        self.image_timestamps = np.asarray(
            [self._timestamp_from_path(path, idx) for idx, path in enumerate(self.color_paths)],
            dtype=np.float64,
        )
        self.load_poses(pd)

    def _load_color_paths(self, folder):
        patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
        image_dirs = [folder, os.path.join(folder, "rgb")]
        paths = []
        for image_dir in image_dirs:
            for pattern in patterns:
                paths.extend(glob.glob(os.path.join(image_dir, pattern)))
                paths.extend(glob.glob(os.path.join(image_dir, pattern.upper())))
        return sorted(set(paths), key=self._image_sort_key)

    def _image_sort_key(self, path):
        stem = os.path.splitext(os.path.basename(path))[0]
        frame_id = stem.split("_", 1)[0]
        try:
            return (0, int(frame_id), stem)
        except ValueError:
            return (1, stem)

    def _frame_id_from_path(self, path):
        stem = os.path.splitext(os.path.basename(path))[0]
        return stem.split("_", 1)[0]

    def _timestamp_from_path(self, path, fallback):
        stem = os.path.splitext(os.path.basename(path))[0]
        if "_" in stem:
            token = stem.rsplit("_", 1)[-1]
            try:
                return float(token)
            except ValueError:
                pass
        return float(fallback)

    def load_poses(self, pd):
        gt_pose_path = os.path.join(self.gt_pose_root, f"{self.scene}.parquet")
        if not os.path.exists(gt_pose_path):
            raise FileNotFoundError(f"Missing railway GT pose parquet: {gt_pose_path}")

        gt_df = pd.read_parquet(gt_pose_path)
        required = ["timestamp", "t_x", "t_y", "t_z", "r_x", "r_y", "r_z", "r_w"]
        missing = [col for col in required if col not in gt_df.columns]
        if missing:
            raise ValueError(f"{gt_pose_path} is missing required columns: {missing}")

        gt_timestamps = gt_df["timestamp"].to_numpy(dtype=np.float64)
        order = np.argsort(gt_timestamps)
        sorted_timestamps = gt_timestamps[order]

        self.poses = []
        self.frames = []
        self.gt_pose_indices = []
        self.gt_pose_time_errors = []
        self.gt_pose_path = gt_pose_path
        first_inv = None

        for i, image_timestamp in enumerate(self.image_timestamps):
            pos = int(np.searchsorted(sorted_timestamps, image_timestamp))
            candidates = []
            if pos < len(sorted_timestamps):
                candidates.append(pos)
            if pos > 0:
                candidates.append(pos - 1)
            if not candidates:
                raise ValueError(f"No GT pose candidate for image timestamp {image_timestamp}")

            best_pos = min(candidates, key=lambda idx: abs(sorted_timestamps[idx] - image_timestamp))
            dt = float(abs(sorted_timestamps[best_pos] - image_timestamp))
            if dt > self.gt_pose_tolerance_sec:
                raise ValueError(
                    f"No GT pose within {self.gt_pose_tolerance_sec}s for image timestamp "
                    f"{image_timestamp}; nearest dt={dt}"
                )

            gt_idx = int(order[best_pos])
            row = gt_df.iloc[gt_idx]
            c2w_abs = np.eye(4, dtype=np.float64)
            c2w_abs[:3, :3] = R.from_quat(
                [row["r_x"], row["r_y"], row["r_z"], row["r_w"]]
            ).as_matrix()
            c2w_abs[:3, 3] = [row["t_x"], row["t_y"], row["t_z"]]
            if first_inv is None:
                first_inv = np.linalg.inv(c2w_abs)
            c2w = first_inv @ c2w_abs
            w2c = np.linalg.inv(c2w)

            self.poses.append(w2c)
            self.gt_pose_indices.append(gt_idx)
            self.gt_pose_time_errors.append(dt)
            self.frames.append({
                "file_path": self.color_paths[i],
                "depth_path": self.color_paths[i],
                "mono_depth_path": self.color_paths[i],
                "transform_matrix": c2w.tolist(),
            })

class KITTIParser:
    def __init__(self, input_folder, config):
        self.input_folder = input_folder
        self.begin = config["Dataset"]["begin"]
        self.end = config["Dataset"]["end"] 
        self.color_paths = sorted(glob.glob(f"{self.input_folder}/rgb/*.png"))[self.begin:self.end]
        self.depth_paths = sorted(glob.glob(f"{self.input_folder}/rgb/*.png"))[self.begin:self.end]
        self.mono_depth_paths = sorted(glob.glob(f"{self.input_folder}/rgb/*.png"))[self.begin:self.end]
        self.n_img = len(self.color_paths)
        self.load_poses(f"{self.input_folder}/gt/*.txt")

    def load_poses(self, path):
        self.poses = []
        self.frames = []
        pose_files = sorted(glob.glob(path))[self.begin:self.end]
        init_trans = np.loadtxt(pose_files[0], delimiter=' ').reshape(4, 4)[:3,3]

        for i in range(self.n_img):
            pose = np.loadtxt(pose_files[i], delimiter=' ').reshape(4, 4)
            pose[:3,3] = pose[:3,3] - init_trans
            inv_pose = np.linalg.inv(pose)  
            self.poses.append(inv_pose)     
            frame = {
                "file_path": self.color_paths[i],
                "depth_path": self.color_paths[i],
                "mono_depth_path": self.color_paths[i],
                "transform_matrix": pose.tolist(),      
            }
            self.frames.append(frame)

class WaymoParser:
    def __init__(self, input_folder):
        self.input_folder = input_folder
        self.color_paths = sorted(glob.glob(f"{self.input_folder}/rgb/*.png"))
        self.depth_paths = sorted(glob.glob(f"{self.input_folder}/depth/*.png"))
        self.mono_depth_paths = sorted(glob.glob(f"{self.input_folder}/mono_depth/*.png"))
        self.n_img = len(self.color_paths)
        self.load_poses(f"{self.input_folder}/gt/*.txt")

    def load_poses(self, path):
        self.poses = []
        self.frames = []
        pose_files = sorted(glob.glob(path))

        for i in range(self.n_img):
            pose = np.loadtxt(pose_files[i], delimiter=' ').reshape(4, 4)
            inv_pose = np.linalg.inv(pose)  
            self.poses.append(inv_pose)     
            frame = {
                "file_path": self.color_paths[i],
                "depth_path": self.depth_paths[i],
                "mono_depth_path": self.mono_depth_paths[i],
                "transform_matrix": pose.tolist(),      
            }
            self.frames.append(frame)

class ReplicaParser:
    def __init__(self, input_folder):
        self.input_folder = input_folder
        self.color_paths = sorted(glob.glob(f"{self.input_folder}/results/frame*.jpg"))
        self.depth_paths = sorted(glob.glob(f"{self.input_folder}/results/depth*.png"))
        self.mono_depth_paths = sorted(glob.glob(f"{self.input_folder}/results/mono*.png"))
        self.n_img = len(self.color_paths)
        self.load_poses(f"{self.input_folder}traj.txt")

    def load_poses(self, path):
        self.poses = []
        with open(path, "r") as f:
            lines = f.readlines()

        frames = []
        for i in range(self.n_img):
            line = lines[i]
            pose = np.array(list(map(float, line.split()))).reshape(4, 4)
            pose = np.linalg.inv(pose)
            self.poses.append(pose)
            frame = {
                "file_path": self.color_paths[i],
                "depth_path": self.depth_paths[i],
                "mono_depth_path": self.mono_depth_paths[i],
                "transform_matrix": pose.tolist(),
            }

            frames.append(frame)
        self.frames = frames


class TUMParser:
    def __init__(self, input_folder):   
        self.input_folder = input_folder
        self.load_poses(self.input_folder, frame_rate=32)
        self.n_img = len(self.color_paths)

    def parse_list(self, filepath, skiprows=0):
        data = np.loadtxt(filepath, delimiter=" ", dtype=np.unicode_, skiprows=skiprows)
        return data

    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
        associations = []
        for i, t in enumerate(tstamp_image):       
            if tstamp_pose is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                if np.abs(tstamp_depth[j] - t) < max_dt:
                    associations.append((i, j))

            else:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))

                if (np.abs(tstamp_depth[j] - t) < max_dt) and (
                    np.abs(tstamp_pose[k] - t) < max_dt
                ):
                    associations.append((i, j, k))

        return associations

    def load_poses(self, datapath, frame_rate=-1):
        if os.path.isfile(os.path.join(datapath, "groundtruth.txt")):
            pose_list = os.path.join(datapath, "groundtruth.txt")
        elif os.path.isfile(os.path.join(datapath, "pose.txt")):
            pose_list = os.path.join(datapath, "pose.txt")

        image_list = os.path.join(datapath, "rgb.txt")
        depth_list = os.path.join(datapath, "depth.txt")
        mono_depth_list = os.path.join(datapath, "mono_depth.txt")

        image_data = self.parse_list(image_list)
        depth_data = self.parse_list(depth_list)
        mono_depth_data = self.parse_list(mono_depth_list)
        pose_data = self.parse_list(pose_list, skiprows=1)
        pose_vecs = pose_data[:, 0:].astype(np.float64)

        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64)
        associations = self.associate_frames(tstamp_image, tstamp_depth, tstamp_pose)
        print("标号:", tstamp_image[471])

        indicies = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indicies[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indicies += [i]

        self.color_paths, self.poses, self.depth_paths, self.frames, self.mono_depth_paths = [], [], [], [], []

        for ix in indicies:
            (i, j, k) = associations[ix]
            self.color_paths += [os.path.join(datapath, image_data[i, 1])]
            self.depth_paths += [os.path.join(datapath, depth_data[j, 1])]
            self.mono_depth_paths += [os.path.join(datapath, mono_depth_data[i, 1])]

            quat = pose_vecs[k][4:]     
            trans = pose_vecs[k][1:4]   
            T = trimesh.transformations.quaternion_matrix(np.roll(quat, 1)) 
            T[:3, 3] = trans
            self.poses += [np.linalg.inv(T)]   

            frame = {
                "file_path": str(os.path.join(datapath, image_data[i, 1])),
                "depth_path": str(os.path.join(datapath, depth_data[j, 1])),
                "transform_matrix": (np.linalg.inv(T)).tolist(),
                "mono_depth_path": str(os.path.join(datapath, mono_depth_data[i, 1]))
            }

            self.frames.append(frame)

##=================================Define data base class==================================
class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, args, path, config):
        self.args = args
        self.path = path
        self.config = config
        self.device = config.get("Dataset", {}).get("device", "cuda:0")
        self.dtype = torch.float32
        self.num_imgs = 999999

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        pass

class MonocularDataset(BaseDataset):
    def __init__(self, args, path, config):     
        super().__init__(args, path, config)
        calibration = config["Dataset"]["Calibration"]
        # Camera prameters
        self.fx = calibration["fx"]
        self.fy = calibration["fy"]    
        self.cx = calibration["cx"]
        self.cy = calibration["cy"]   
        self.width = calibration["width"]
        self.height = calibration["height"]
        self.fovx = focal2fov(self.fx, self.width)
        self.fovy = focal2fov(self.fy, self.height)
        self.K = np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )                               
        # distortion parameters
        self.disorted = calibration["distorted"] 
        self.dist_coeffs = np.array(
            [
                calibration["k1"],
                calibration["k2"],
                calibration["p1"],
                calibration["p2"],
                calibration["k3"],
            ]
        )
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(  
            self.K,
            self.dist_coeffs,
            np.eye(3),
            self.K,
            (self.width, self.height),
            cv2.CV_32FC1,
        )
        # depth parameters
        self.has_depth = True if "depth_scale" in calibration.keys() else False
        self.depth_scale = calibration["depth_scale"] if self.has_depth else None

        # Default scene scale  
        nerf_normalization_radius = 5
        self.scene_info = {
            "nerf_normalization": {
                "radius": nerf_normalization_radius,
                "translation": np.zeros(3),
            },
        }

    def load_image(self, image_path):
        image = Image.open(image_path)
        image_array = np.array(image)

        # Check if the image is RGB (3 channels); if so, extract the first channel
        if len(image_array.shape) == 3:  
            return image_array[:, :, 0]  
        else:  
            return image_array  

    def __getitem__(self, idx):  
        color_path = self.color_paths[idx]
        pose = self.poses[idx]

        image = np.array(Image.open(color_path))
        depth = None
        mono_depth = self.load_image(color_path).astype(np.float32)

        if self.disorted:
            image = cv2.remap(image, self.map1x, self.map1y, cv2.INTER_LINEAR)  

        if self.has_depth:
            depth_path = self.depth_paths[idx]
            depth = self.load_image(depth_path) / self.depth_scale  
            mono_depth_path = self.mono_depth_paths[idx]
            mono_depth = self.load_image(mono_depth_path) / (self.depth_scale*5)

        image = (
            torch.from_numpy(image / 255.0)
            .clamp(0.0, 1.0)
            .permute(2, 0, 1)
            .to(device=self.device, dtype=self.dtype)
        )
        pose = torch.from_numpy(pose).to(device=self.device)
        return image, depth, pose, mono_depth

##=====================================# Define dataset class for specific dataset======================================
class dl3dvDataset(MonocularDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]
        
        parser = dl3dvParser(dataset_path, config)

        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.depth_paths = parser.color_paths  
        self.mono_depth_paths = parser.color_paths  
        self.poses = parser.poses  

class KITTIDataset(MonocularDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]
        parser = KITTIParser(dataset_path,config)
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.depth_paths = parser.depth_paths
        self.mono_depth_paths = parser.mono_depth_paths
        self.poses = parser.poses      

class WaymoDataset(MonocularDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]
        parser = WaymoParser(dataset_path)
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.depth_paths = parser.depth_paths
        self.mono_depth_paths = parser.mono_depth_paths
        self.poses = parser.poses       

class TUMDataset(MonocularDataset):  
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]
        parser = TUMParser(dataset_path)
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.depth_paths = parser.depth_paths
        self.poses = parser.poses
        self.mono_depth_paths = parser.mono_depth_paths

class RailwayDataset(MonocularDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_cfg = config["Dataset"]
        dataset_path = dataset_cfg.get("dataset_path")
        if not dataset_path:
            dataset_root = dataset_cfg["dataset_root"]
            scene = dataset_cfg["scene"]
            dataset_path = os.path.join(dataset_root, scene)
        parser = RailwayParser(dataset_path, config)
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.depth_paths = parser.depth_paths
        self.mono_depth_paths = parser.mono_depth_paths
        self.poses = parser.poses
        self.frame_ids = parser.frame_ids
        self.image_names = parser.image_names
        self.image_timestamps = parser.image_timestamps
        self.gt_pose_indices = np.asarray(parser.gt_pose_indices, dtype=np.int64)
        self.gt_pose_time_errors = np.asarray(parser.gt_pose_time_errors, dtype=np.float64)
        self.gt_pose_path = parser.gt_pose_path
        self.scene = parser.scene
        self.has_gt_depth = False
        self.original_width = dataset_cfg["OriginalCalibration"]["width"]
        self.original_height = dataset_cfg["OriginalCalibration"]["height"]
        self.original_K = np.array(
            [
                [dataset_cfg["OriginalCalibration"]["fx"], 0.0, dataset_cfg["OriginalCalibration"]["cx"]],
                [0.0, dataset_cfg["OriginalCalibration"]["fy"], dataset_cfg["OriginalCalibration"]["cy"]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self.original_dist_coeffs = np.array(
            [
                dataset_cfg["OriginalCalibration"]["k1"],
                dataset_cfg["OriginalCalibration"]["k2"],
                dataset_cfg["OriginalCalibration"]["p1"],
                dataset_cfg["OriginalCalibration"]["p2"],
                dataset_cfg["OriginalCalibration"]["k3"],
            ],
            dtype=np.float64,
        )

    def __getitem__(self, idx):
        color_path = self.color_paths[idx]
        pose = self.poses[idx]
        image = np.array(Image.open(color_path).convert("RGB"))
        if self.config["Dataset"].get("undistort", True):
            image = cv2.undistort(image, self.original_K, self.original_dist_coeffs)
        if image.shape[1] != self.width or image.shape[0] != self.height:
            image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)

        mono_depth = image[:, :, 0].astype(np.float32) / 255.0
        depth = mono_depth.copy()
        image = (
            torch.from_numpy(image / 255.0)
            .clamp(0.0, 1.0)
            .permute(2, 0, 1)
            .to(device=self.device, dtype=self.dtype)
        )
        pose = torch.from_numpy(pose).to(device=self.device, dtype=self.dtype)
        return image, depth, pose, mono_depth

class ReplicaDataset(MonocularDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        dataset_path = config["Dataset"]["dataset_path"]
        parser = ReplicaParser(dataset_path)
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.depth_paths = parser.depth_paths
        self.mono_depth_paths = parser.mono_depth_paths
        self.poses = parser.poses

def load_dataset(args, path, config):
    if config["Dataset"]["type"] == "tum":
        return TUMDataset(args, path, config)
    elif config["Dataset"]["type"] == "replica":
        return ReplicaDataset(args, path, config)
    elif config["Dataset"]["type"] == "waymo":
        return WaymoDataset(args, path, config)
    elif config["Dataset"]["type"] == "KITTI":
        return KITTIDataset(args, path, config)
    elif config["Dataset"]["type"] == "dl3dv":
        return dl3dvDataset(args, path, config)
    elif config["Dataset"]["type"] == "railway":
        return RailwayDataset(args, path, config)
    else:
        raise ValueError("Unknown dataset type")
