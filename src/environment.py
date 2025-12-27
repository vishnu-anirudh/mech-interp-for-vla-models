"""
PyBullet robotics simulation environment for generating training data.
"""

import numpy as np
import pybullet as p
import pybullet_data
from PIL import Image


class RoboticsEnv:
    """
    PyBullet simulation environment for robotics scenarios.
    Generates images of objects (fragile vs rigid) for vision-language model training.
    """

    def __init__(self, gui=False):
        """
        Initialize the PyBullet simulation environment.

        Args:
            gui: Whether to show the GUI (default: False for headless operation)
        """
        mode = p.GUI if gui else p.DIRECT
        try:
            p.connect(mode)
        except Exception:
            p.connect(p.DIRECT)  # Fallback if GUI fails

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        self.width = 384  # Higher res for VLM
        self.height = 384
        self.reset()

    def reset(self):
        """Reset the simulation to initial state."""
        p.resetSimulation()
        p.loadURDF("plane.urdf")
        self.robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

    def spawn_object(self, is_fragile: bool):
        """
        Spawn an object in the simulation.

        Args:
            is_fragile: If True, spawns a red (fragile) object, else blue (rigid)
        """
        color = [0.9, 0.1, 0.1, 1] if is_fragile else [0.1, 0.1, 0.9, 1]  # Red=Fragile, Blue=Rigid
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.05)
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=color)
        self.obj_id = p.createMultiBody(1, col, vis, [0.5, 0, 0.05])

    def spawn_object_decorrelated(self, is_fragile: bool, color: str = "red"):
        """
        Spawn an object with decorrelated appearance and physics.
        Addresses Clever Hans confounder.

        Args:
            is_fragile: Physical property (True = fragile, False = rigid)
            color: Visual appearance ('red' or 'blue')
        """
        color_map = {
            "red": [0.9, 0.1, 0.1, 1],
            "blue": [0.1, 0.1, 0.9, 1],
            "green": [0.1, 0.9, 0.1, 1],  # Additional option
            "yellow": [0.9, 0.9, 0.1, 1],  # Additional option
        }

        rgba_color = color_map.get(color.lower(), color_map["red"])
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.05)
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=rgba_color)
        self.obj_id = p.createMultiBody(1, col, vis, [0.5, 0, 0.05])

    def get_image(self) -> Image.Image:
        """
        Capture an image from the simulation.

        Returns:
            PIL Image of the current simulation state
        """
        view_matrix = p.computeViewMatrix([0.5, -0.5, 0.5], [0.5, 0, 0], [0, 0, 1])
        proj_matrix = p.computeProjectionMatrixFOV(60, 1.0, 0.1, 10.0)

        w, h, rgb, _, _ = p.getCameraImage(self.width, self.height, view_matrix, proj_matrix)
        rgb_array = np.reshape(rgb, (h, w, 4))[:, :, :3]
        # Convert to uint8 and ensure values are in valid range [0, 255]
        rgb_array = np.clip(rgb_array, 0, 255).astype(np.uint8)
        return Image.fromarray(rgb_array)
