# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Simple script to control a robot from teleoperation with a graphical compliance slider.

Requires: pip install 'lerobot[hardware]'
"""

import logging
import time
import threading
import tkinter as tk
from dataclasses import asdict, dataclass
from pprint import pformat

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq import ZMQCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_openarm_follower,
    bi_rebot_b601_follower,
    bi_so_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    openarm_follower,
    reachy2,
    rebot_b601_follower,
    so_follower,
    unitree_g1 as unitree_g1_robot,
)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_openarm_leader,
    bi_openarm_mini,
    bi_rebot_102_leader,
    bi_so_leader,
    gamepad,
    homunculus,
    keyboard,
    koch_leader,
    make_teleoperator_from_config,
    omx_leader,
    openarm_leader,
    openarm_mini,
    reachy2_teleoperator,
    rebot_102_leader,
    so_leader,
    unitree_g1,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, move_cursor_up
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data, shutdown_rerun


@dataclass
class TeleoperateConfig:
    teleop: TeleoperatorConfig
    robot: RobotConfig
    fps: int = 60
    teleop_time_s: float | None = None
    display_data: bool = False
    display_ip: str | None = None
    display_port: int | None = None
    display_compressed_images: bool = False


# Global variable to safely communicate between the GUI thread and Teleop thread
shared_compliance_margin = 2


def launch_gui():
    """Builds and manages the graphical interface window in a background thread."""
    global shared_compliance_margin

    root = tk.Tk()
    root.title("SO-101 Compliance Tuner")
    root.geometry("400x180")
    root.resizable(False, False)

    # Styling elements
    root.configure(bg="#2c3e50")
    
    label_title = tk.Label(
        root, 
        text="Whiteboard Compliance Tuning", 
        font=("Helvetica", 14, "bold"), 
        fg="#ecf0f1", 
        bg="#2c3e50"
    )
    label_title.pack(pady=10)

    label_desc = tk.Label(
        root,
        text="Slide right to soften pen pressure (increase margin).\nSlide left to stiffen mechanical rigidity.",
        font=("Helvetica", 9),
        fg="#bdc3c7",
        bg="#2c3e50"
    )
    label_desc.pack()

    def on_slider_change(val):
        global shared_compliance_margin
        shared_compliance_margin = int(val)

    slider = tk.Scale(
        root, 
        from_=0, 
        to=15, 
        orient="horizontal", 
        command=on_slider_change,
        bg="#34495e",
        fg="#ecf0f1",
        highlightbackground="#2c3e50",
        troughcolor="#1abc9c",
        length=320
    )
    slider.set(shared_compliance_margin)
    slider.pack(pady=15)

    root.mainloop()


def teleop_loop(
    teleop: Teleoperator,
    robot: Robot,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
    display_data: bool = False,
    duration: float | None = None,
    display_compressed_images: bool = False,
):
    global shared_compliance_margin
    display_len = max(len(key) for key in robot.action_features)
    start = time.perf_counter()

    # --- Compliance Configuration ---
    PITCH_SERVO_IDS = [2, 3]  # Replace with your actual pitch/elbow IDs if different
    REG_CW_COMPLIANCE_MARGIN = 28
    REG_CCW_COMPLIANCE_MARGIN = 29
    
    # Keep track of the last pushed value to avoid spamming the serial bus unnecessarily
    last_pushed_margin = -1

    # Launch the Tkinter GUI thread
    gui_thread = threading.Thread(target=launch_gui, daemon=True)
    gui_thread.start()

    while True:
        loop_start = time.perf_counter()

        # Get robot observation
        obs = robot.get_observation()

        if robot.name == "unitree_g1":
            teleop.send_feedback(obs)

        # Get teleop action
        raw_action = teleop.get_action()

        # Process teleop action through pipeline
        teleop_action = teleop_action_processor((raw_action, obs))

        # Process action for robot through pipeline
        robot_action_to_send = robot_action_processor((teleop_action, obs))

        # Send processed action to robot
        _ = robot.send_action(robot_action_to_send)

        # --- Check GUI state and push to hardware ---
        snapshot_margin = shared_compliance_margin
        if snapshot_margin != last_pushed_margin:
            follower_bus = None
            if hasattr(robot, "follower") and hasattr(robot.follower, "follower_bus"):
                follower_bus = robot.follower.follower_bus
            elif hasattr(robot, "motors") and "follower" in robot.motors:
                follower_bus = robot.motors["follower"]

            if follower_bus is not None:
                for servo_id in PITCH_SERVO_IDS:
                    follower_bus.write_register(servo_id, REG_CW_COMPLIANCE_MARGIN, snapshot_margin)
                    follower_bus.write_register(servo_id, REG_CCW_COMPLIANCE_MARGIN, snapshot_margin)
                last_pushed_margin = snapshot_margin
            else:
                print("\r[Error] Could not find follower motor bus link.                  ", end="")

        if display_data:
            obs_transition = robot_observation_processor(obs)
            log_rerun_data(
                observation=obs_transition,
                action=teleop_action,
                compress_images=display_compressed_images,
            )

            print("\n" + "-" * (display_len + 10))
            print(f"{'NAME':<{display_len}} | {'NORM':>7}")
            for motor, value in robot_action_to_send.items():
                print(f"{motor:<{display_len}} | {value:>7.2f}")
            move_cursor_up(len(robot_action_to_send) + 3)

        dt_s = time.perf_counter() - loop_start
        precise_sleep(max(1 / fps - dt_s, 0.0))
        loop_s = time.perf_counter() - loop_start
        
        if not display_data:
            print(f"\rTeleop loop time: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz) | Hardware Margin Set: {last_pushed_margin}   ", end="")
        else:
            print(f"Teleop loop time: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")
            move_cursor_up(1)

        if duration is not None and time.perf_counter() - start >= duration:
            return


@parser.wrap()
def teleoperate(cfg: TeleoperateConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(session_name="teleoperation", ip=cfg.display_ip, port=cfg.display_port)
    display_compressed_images = (
        True
        if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
        else cfg.display_compressed_images
    )

    teleop = make_teleoperator_from_config(cfg.teleop)
    robot = make_robot_from_config(cfg.robot)
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    teleop.connect()
    robot.connect()

    try:
        teleop_loop(
            teleop=teleop,
            robot=robot,
            fps=cfg.fps,
            display_data=cfg.display_data,
            duration=cfg.teleop_time_s,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            display_compressed_images=display_compressed_images,
        )
    except KeyboardInterrupt:
        pass
    finally:
        if cfg.display_data:
            shutdown_rerun()
        teleop.disconnect()
        robot.disconnect()


def main():
    register_third_party_plugins()
    teleoperate()


if __name__ == "__main__":
    main()