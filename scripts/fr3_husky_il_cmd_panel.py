#!/usr/bin/env python3
"""
ROS 2 action client node for Vive-controller-based teleoperation 
and Imitation Learning policies of FR3-Husky.

A PyQt5 GUI lets the operator set goal parameters and send / cancel the
ViveTracker goal at any time. rclpy spins in a background thread so the
Qt event loop owns the main thread.

Subscribes to:
    lhand_joy   (sensor_msgs/Joy) - left Vive controller buttons/axes
    rhand_joy   (sensor_msgs/Joy) - right Vive controller buttons/axes

Publishes:
    lerobot/start_episode   (std_msgs/Bool) - starts data collection / inference
    lerobot/save_episode    (std_msgs/Bool) - save / discard current episode data

Sends goals to:
    fr3_vive_tracker        (fr3_husky_msgs/action/ViveTracker)
    fr3_task_space_policy   (fr3_husky_msgs/action/TaskSpacePolicy)
    fr3_move_to_joint       (fr3_husky_msgs/action/MoveToJoint)

Button layout (Vive controller):
    axes[0]   trigger   (0 ~ 1)
    axes[1]   grip      (0 ~ 1)
    axes[2]   joy_x     (-1 ~ 1)
    axes[3]   joy_y     (-1 ~ 1)

    buttons[0]  trigger
    buttons[1]  grip
    buttons[2]  A/X
    buttons[3]  B/Y
    buttons[4]  joystick click
"""

import sys
import threading

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle
from std_msgs.msg import Bool
from sensor_msgs.msg import Joy
from fr3_husky_msgs.action import ViveTracker
from fr3_husky_msgs.action import TaskSpacePolicy
from fr3_husky_msgs.action import MoveToJoint

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QComboBox, QLineEdit, QCheckBox, QDoubleSpinBox,
    QPushButton, QGroupBox,
)
from PyQt5.QtCore import Qt, pyqtSignal, QObject


# ── Constants ─────────────────────────────────────────────────────────────

# Joy index
IDX_TRIGGER_AXES = 0
IDX_GRIP_AXES    = 1
IDX_JOY_X_AXES   = 2
IDX_JOY_Y_AXES   = 3

IDX_TRIGGER_BUTTON = 0
IDX_GRIP_BUTTON    = 1
IDX_A_BUTTON       = 2
IDX_B_BUTTON       = 3
IDX_JOY_BUTTON     = 4

_MODE_LABELS = ["0: CLIK", "1: OSF", "2: QPIK", "3: QPID"]

# Ready-pose joint positions [rad] for left / right FR3 (joints 1–7)
_LEFT_READY_POSE  = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
_RIGHT_READY_POSE = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
import math
_DEG2RAD = math.pi / 180
_RAD2DEG = 180 / math.pi


# ── Qt signal bridge (thread-safe node → GUI updates) ─────────────────────

class _NodeSignals(QObject):
    status_changed           = pyqtSignal(str)  # teleoperation status
    save_count_changed       = pyqtSignal(int)
    inference_status_changed = pyqtSignal(str)  # TaskSpacePolicy status
    move_status_changed      = pyqtSignal(str)  # MoveToJoint status


# ── ROS node ──────────────────────────────────────────────────────────────

class FR3HuskyILCommand(Node):

    def __init__(self, signals: _NodeSignals) -> None:
        super().__init__("fr3_husky_il_command")
        self._signals = signals

        # ── Parameters ────────────────────────────────────────────────────
        self.declare_parameter("mode", 2)
        self.declare_parameter("left_controller_ee_name",  "left_fr3_hand_tcp")
        self.declare_parameter("right_controller_ee_name", "right_fr3_hand_tcp")
        self.declare_parameter("move_orientation", True)
        self.declare_parameter("controller_pos_multiplier", 0.8)
        self.declare_parameter("controller_ori_multiplier", 0.8)

        # ── Action clients ────────────────────────────────────────────────
        self._vive_client      = ActionClient(self, ViveTracker,      "fr3_vive_tracker")
        self._policy_client    = ActionClient(self, TaskSpacePolicy,  "fr3_task_space_policy")
        self._move_joint_client = ActionClient(self, MoveToJoint,     "fr3_move_to_joint")

        self._vive_goal_handle:   ClientGoalHandle | None = None
        self._policy_goal_handle: ClientGoalHandle | None = None
        self._lock = threading.Lock()

        # ── Previous button states ────────────────────────────────────────
        self._prev_l_buttons: list[int] = [0] * 5
        self._prev_r_buttons: list[int] = [0] * 5

        # ── Subscriptions ─────────────────────────────────────────────────
        self.create_subscription(Joy, "lhand_joy", self._lhand_joy_callback, 10)
        self.create_subscription(Joy, "rhand_joy", self._rhand_joy_callback, 10)
        self.lhand_grip_pressed = False
        self.rhand_grip_pressed = False
        self._prev_mouse_mode_enabled = False

        # ── Publishers ────────────────────────────────────────────────────
        self.lerobot_start_episode_publisher = self.create_publisher(
            Bool, "lerobot/start_episode", 1)
        self.lerobot_save_episode_publisher = self.create_publisher(
            Bool, "lerobot/save_episode", 1)

        self._save_count = 0

        # ── Timer ─────────────────────────────────────────────────────────
        self._pub_timer = self.create_timer(0.01, self._publish_mouse_mode)

        self.get_logger().info("Vive teleoperation client ready.")
        self._signals.status_changed.emit("Ready — no goal sent yet.")

    # ── Teleoperation goal control ────────────────────────────────────────

    def send_goal(
        self,
        mode: int,
        left_ee: str,
        right_ee: str,
        move_ori: bool,
        pos_mult: float,
        ori_mult: float,
    ) -> None:
        if not self._vive_client.wait_for_server(timeout_sec=5.0):
            msg = "ViveTracker action server not available."
            self.get_logger().error(msg)
            self._signals.status_changed.emit(f"Error: {msg}")
            return

        goal = ViveTracker.Goal()
        goal.mode                      = mode
        goal.left_controller_ee_name   = left_ee
        goal.right_controller_ee_name  = right_ee
        goal.move_orientation          = move_ori
        goal.controller_pos_multiplier = pos_mult
        goal.controller_ori_multiplier = ori_mult

        self.get_logger().info(
            f"Sending ViveTracker goal: mode={mode}, left_ee={left_ee}, "
            f"right_ee={right_ee}, move_ori={move_ori}, "
            f"pos_mult={pos_mult:.2f}, ori_mult={ori_mult:.2f}"
        )
        self._signals.status_changed.emit("Sending goal…")

        future = self._vive_client.send_goal_async(
            goal, feedback_callback=self._vive_feedback_callback)
        future.add_done_callback(self._vive_goal_response_callback)

    def cancel_goal(self) -> None:
        with self._lock:
            handle = self._vive_goal_handle

        if handle is None:
            self._signals.status_changed.emit("No active goal to cancel.")
            return

        self.get_logger().info("Cancelling ViveTracker goal…")
        self._signals.status_changed.emit("Cancelling goal…")
        future = handle.cancel_goal_async()
        future.add_done_callback(
            lambda _: self._signals.status_changed.emit("Goal cancel requested."))

    # ── Teleoperation action callbacks ────────────────────────────────────

    def _vive_goal_response_callback(self, future) -> None:
        goal_handle: ClientGoalHandle = future.result()
        if not goal_handle.accepted:
            msg = "ViveTracker goal rejected."
            self.get_logger().error(msg)
            self._signals.status_changed.emit(f"Error: {msg}")
            return

        self.get_logger().info("ViveTracker goal accepted — teleoperation running.")
        self._enable_teleop_pub = True
        self._signals.status_changed.emit("Goal accepted — teleoperation running.")
        with self._lock:
            self._vive_goal_handle = goal_handle

        goal_handle.get_result_async().add_done_callback(self._vive_result_callback)

    def _vive_feedback_callback(self, feedback_msg) -> None:
        fb: ViveTracker.Feedback = feedback_msg.feedback
        if not fb.is_qp_solved:
            self.get_logger().warn(f"QP not solved | {fb.time_verbose}")

    def _vive_result_callback(self, future) -> None:
        result = future.result().result
        status = future.result().status
        self._enable_teleop_pub = False

        with self._lock:
            self._vive_goal_handle = None

        if result.is_completed:
            msg = f"Goal completed successfully (status={status})."
            self.get_logger().info(msg)
        else:
            msg = f"Goal ended without completion (status={status})."
            self.get_logger().warn(msg)

        self._signals.status_changed.emit(msg)

    # ── TaskSpacePolicy goal control ──────────────────────────────────────

    def send_policy_goal(self, mode: int, left_ee: str, right_ee: str) -> None:
        if not self._policy_client.wait_for_server(timeout_sec=5.0):
            msg = "TaskSpacePolicy action server not available."
            self.get_logger().error(msg)
            self._signals.inference_status_changed.emit(f"Error: {msg}")
            return

        goal = TaskSpacePolicy.Goal()
        goal.mode              = mode
        goal.left_fr3_ee_name  = left_ee
        goal.right_fr3_ee_name = right_ee

        self.get_logger().info(
            f"Sending TaskSpacePolicy goal: mode={mode}, "
            f"left_ee={left_ee}, right_ee={right_ee}"
        )
        self._signals.inference_status_changed.emit("Sending policy goal…")

        future = self._policy_client.send_goal_async(
            goal, feedback_callback=self._policy_feedback_callback)
        future.add_done_callback(self._policy_goal_response_callback)

    def cancel_policy_goal(self) -> None:
        self._publish_start_episode(False)
        with self._lock:
            handle = self._policy_goal_handle

        if handle is None:
            self._signals.inference_status_changed.emit("No active policy goal to cancel.")
            return

        self.get_logger().info("Cancelling TaskSpacePolicy goal…")
        self._signals.inference_status_changed.emit("Cancelling policy goal…")
        future = handle.cancel_goal_async()
        future.add_done_callback(
            lambda _: self._signals.inference_status_changed.emit("Policy goal cancel requested."))

    # ── TaskSpacePolicy action callbacks ──────────────────────────────────

    def _policy_goal_response_callback(self, future) -> None:
        goal_handle: ClientGoalHandle = future.result()
        if not goal_handle.accepted:
            msg = "TaskSpacePolicy goal rejected."
            self.get_logger().error(msg)
            self._signals.inference_status_changed.emit(f"Error: {msg}")
            return

        self.get_logger().info("TaskSpacePolicy goal accepted — inference running.")
        self._publish_start_episode(True)
        self._signals.inference_status_changed.emit("Policy goal accepted — inference running.")
        with self._lock:
            self._policy_goal_handle = goal_handle

        goal_handle.get_result_async().add_done_callback(self._policy_result_callback)

    def _policy_feedback_callback(self, feedback_msg) -> None:
        fb: TaskSpacePolicy.Feedback = feedback_msg.feedback
        if not fb.is_qp_solved:
            self.get_logger().warn(f"[Policy] QP not solved | {fb.time_verbose}")

    def _policy_result_callback(self, future) -> None:
        result = future.result().result
        status = future.result().status

        with self._lock:
            self._policy_goal_handle = None

        if result.is_completed:
            msg = f"Policy goal completed (status={status})."
            self.get_logger().info(msg)
        else:
            msg = f"Policy goal ended without completion (status={status})."
            self.get_logger().warn(msg)

        self._signals.inference_status_changed.emit(msg)

    # ── MoveToJoint goal control ──────────────────────────────────────────

    def send_move_to_joint_goal(
        self,
        joint_names: list[str],
        target_positions: list[float],
        vel_scale: float,
        acc_scale: float,
    ) -> None:
        if not self._move_joint_client.wait_for_server(timeout_sec=5.0):
            msg = "MoveToJoint action server not available."
            self.get_logger().error(msg)
            self._signals.move_status_changed.emit(f"Error: {msg}")
            return

        goal = MoveToJoint.Goal()
        goal.joint_names                    = joint_names
        goal.target_positions               = target_positions
        goal.max_velocity_scaling_factor     = vel_scale
        goal.max_acceleration_scaling_factor = acc_scale

        self.get_logger().info(
            f"Sending MoveToJoint goal: joints={joint_names}, "
            f"positions={target_positions}, vel={vel_scale:.2f}, acc={acc_scale:.2f}"
        )
        self._signals.move_status_changed.emit("Sending MoveToJoint goal…")

        future = self._move_joint_client.send_goal_async(
            goal, feedback_callback=self._move_feedback_callback)
        future.add_done_callback(self._move_goal_response_callback)

    # ── MoveToJoint action callbacks ──────────────────────────────────────

    def _move_goal_response_callback(self, future) -> None:
        goal_handle: ClientGoalHandle = future.result()
        if not goal_handle.accepted:
            msg = "MoveToJoint goal rejected."
            self.get_logger().error(msg)
            self._signals.move_status_changed.emit(f"Error: {msg}")
            return

        self.get_logger().info("MoveToJoint goal accepted.")
        self._signals.move_status_changed.emit("MoveToJoint goal accepted — moving…")
        goal_handle.get_result_async().add_done_callback(self._move_result_callback)

    def _move_feedback_callback(self, feedback_msg) -> None:
        fb: MoveToJoint.Feedback = feedback_msg.feedback
        state_str = "PLANNING" if fb.state == 0 else "EXECUTING"
        self._signals.move_status_changed.emit(
            f"{state_str} — {fb.progress * 100:.0f}%  {fb.status_message}")

    def _move_result_callback(self, future) -> None:
        result = future.result().result
        status = future.result().status

        if result.success:
            msg = f"MoveToJoint completed (status={status})."
            self.get_logger().info(msg)
        else:
            msg = f"MoveToJoint failed: {result.message} (code={result.error_code})."
            self.get_logger().error(msg)

        self._signals.move_status_changed.emit(msg)

    # ── Joy callbacks ─────────────────────────────────────────────────────

    def _lhand_joy_callback(self, msg: Joy) -> None:
        buttons = list(msg.buttons)
        self.lhand_grip_pressed = bool(buttons[IDX_GRIP_BUTTON])
        l_a_pressed = self._rising_edge(buttons, self._prev_l_buttons, IDX_A_BUTTON)
        l_b_pressed = self._rising_edge(buttons, self._prev_l_buttons, IDX_B_BUTTON)

        if l_a_pressed:
            self._publish_save_episode(False)
        if l_b_pressed:
            self._publish_save_episode(True)

        self._prev_l_buttons = buttons

    def _rhand_joy_callback(self, msg: Joy) -> None:
        buttons = list(msg.buttons)
        self.rhand_grip_pressed = bool(buttons[IDX_GRIP_BUTTON])
        r_a_pressed = self._rising_edge(buttons, self._prev_r_buttons, IDX_A_BUTTON)
        r_b_pressed = self._rising_edge(buttons, self._prev_r_buttons, IDX_B_BUTTON)

        if r_a_pressed:
            self._publish_save_episode(False)
        if r_b_pressed:
            self._publish_save_episode(True)

        self._prev_r_buttons = buttons

    def _publish_save_episode(self, data: bool) -> None:
        self.lerobot_save_episode_publisher.publish(Bool(data=data))
        if data:
            self._save_count += 1
            self._signals.save_count_changed.emit(self._save_count)

    # ── Timer callback ────────────────────────────────────────────────────

    def _publish_start_episode(self, data: bool) -> None:
        self.lerobot_start_episode_publisher.publish(Bool(data=data))
    
    def _publish_mouse_mode(self) -> None:
        mouse_mode_enabled = self.lhand_grip_pressed or self.rhand_grip_pressed
        if self._rising_edge(mouse_mode_enabled, self._prev_mouse_mode_enabled):
            self._publish_start_episode(True)
        elif self._falling_edge(mouse_mode_enabled, self._prev_mouse_mode_enabled):
            self._publish_start_episode(False)
        self._prev_mouse_mode_enabled = mouse_mode_enabled

    # ── Utilities ─────────────────────────────────────────────────────────
    
    @staticmethod
    def _rising_edge(curr, prev, idx: int = None) -> bool:
        if idx is not None:
            return bool(curr[idx]) and not bool(prev[idx])
        else:
            return bool(curr) and not bool(prev)
    
    @staticmethod
    def _falling_edge(curr, prev, idx: int = None) -> bool:
        if idx is not None:
            return not bool(curr[idx]) and bool(prev[idx])
        else:
            return not bool(curr) and bool(prev)


# ── PyQt5 GUI ─────────────────────────────────────────────────────────────

class CommandPanelGUI(QWidget):

    def __init__(self, node: FR3HuskyILCommand, signals: _NodeSignals) -> None:
        super().__init__()
        self._node = node
        self._build_ui()
        signals.status_changed.connect(self._on_status_changed)
        signals.save_count_changed.connect(self._on_save_count_changed)
        signals.inference_status_changed.connect(self._on_inference_status_changed)
        signals.move_status_changed.connect(self._on_move_status_changed)
        self.setWindowTitle("FR3 Husky IL Command Panel")

    # ── UI construction ───────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)

        left_col = QVBoxLayout()
        tele_box = self._build_teleoperation_group()
        infer_box = self._build_inference_group()
        left_col.addWidget(tele_box)
        left_col.addStretch()
        left_col.addWidget(infer_box)
        root.addLayout(left_col)

        move_box = self._build_move_to_joint_group()
        root.addWidget(move_box)

    # ── Teleoperation group ───────────────────────────────────────────────

    def _build_teleoperation_group(self) -> QGroupBox:
        box = QGroupBox("Teleoperation")
        layout = QVBoxLayout(box)

        form = QFormLayout()

        self._mode_combo = QComboBox()
        self._mode_combo.addItems(_MODE_LABELS)
        self._mode_combo.setCurrentIndex(2)
        form.addRow("Mode", self._mode_combo)

        self._left_ee = QLineEdit("left_fr3_hand_tcp")
        form.addRow("L con EE name", self._left_ee)

        self._right_ee = QLineEdit("right_fr3_hand_tcp")
        form.addRow("R con EE name", self._right_ee)

        self._move_ori = QCheckBox()
        self._move_ori.setChecked(True)
        form.addRow("Move orientation", self._move_ori)

        self._pos_mult = QDoubleSpinBox()
        self._pos_mult.setRange(0.0, 5.0)
        self._pos_mult.setSingleStep(0.1)
        self._pos_mult.setValue(0.8)
        form.addRow("Pos multiplier", self._pos_mult)

        self._ori_mult = QDoubleSpinBox()
        self._ori_mult.setRange(0.0, 5.0)
        self._ori_mult.setSingleStep(0.1)
        self._ori_mult.setValue(0.8)
        form.addRow("Ori multiplier", self._ori_mult)

        layout.addLayout(form)

        btn_row = QHBoxLayout()
        self._send_btn = QPushButton("Send Goal")
        self._send_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self._send_btn.clicked.connect(self._on_send)
        btn_row.addWidget(self._send_btn)

        self._cancel_btn = QPushButton("Cancel Goal")
        self._cancel_btn.setStyleSheet("background-color: #f44336; color: white; font-weight: bold;")
        self._cancel_btn.clicked.connect(self._on_cancel)
        btn_row.addWidget(self._cancel_btn)
        layout.addLayout(btn_row)

        counter_row = QHBoxLayout()
        counter_row.addWidget(QLabel("Episodes saved:"))
        self._save_count_label = QLabel("0")
        self._save_count_label.setStyleSheet("font-weight: bold; font-size: 16px;")
        counter_row.addWidget(self._save_count_label)
        counter_row.addStretch()
        layout.addLayout(counter_row)

        self._status_label = QLabel("")
        self._status_label.setAlignment(Qt.AlignCenter)
        self._status_label.setStyleSheet("border: 1px solid #aaa; padding: 4px;")
        layout.addWidget(self._status_label)

        return box

    # ── Inference group ───────────────────────────────────────────────────

    def _build_inference_group(self) -> QGroupBox:
        box = QGroupBox("Inference")
        box.setLayout(QVBoxLayout())
        box.layout().addWidget(self._build_task_space_policy_subgroup())
        return box

    def _build_task_space_policy_subgroup(self) -> QGroupBox:
        policy_box = QGroupBox("TaskSpacePolicy")
        layout = QVBoxLayout(policy_box)
        form = QFormLayout()

        self._policy_mode_combo = QComboBox()
        self._policy_mode_combo.addItems(_MODE_LABELS)
        self._policy_mode_combo.setCurrentIndex(2)
        form.addRow("Mode", self._policy_mode_combo)

        self._policy_left_ee = QLineEdit("left_fr3_hand_tcp")
        form.addRow("Left EE name", self._policy_left_ee)

        self._policy_right_ee = QLineEdit("right_fr3_hand_tcp")
        form.addRow("Right EE name", self._policy_right_ee)

        layout.addLayout(form)

        btn_row = QHBoxLayout()
        self._policy_send_btn = QPushButton("Send Goal")
        self._policy_send_btn.setStyleSheet(
            "background-color: #4CAF50; color: white; font-weight: bold;")
        self._policy_send_btn.clicked.connect(self._on_policy_send)
        btn_row.addWidget(self._policy_send_btn)

        self._policy_cancel_btn = QPushButton("Cancel Goal")
        self._policy_cancel_btn.setStyleSheet(
            "background-color: #f44336; color: white; font-weight: bold;")
        self._policy_cancel_btn.clicked.connect(self._on_policy_cancel)
        btn_row.addWidget(self._policy_cancel_btn)
        layout.addLayout(btn_row)

        self._inference_status_label = QLabel("")
        self._inference_status_label.setAlignment(Qt.AlignCenter)
        self._inference_status_label.setStyleSheet("border: 1px solid #aaa; padding: 4px;")
        layout.addWidget(self._inference_status_label)

        return policy_box

    # ── MoveToJoint group ─────────────────────────────────────────────────

    def _build_move_to_joint_group(self) -> QGroupBox:
        move_box = QGroupBox("Move to Initial Position")
        layout = QVBoxLayout(move_box)

        # ── Per-joint fields (left | right side by side) ──────────────────
        self._joint_edits: dict[str, QLineEdit] = {}
        joints_row = QHBoxLayout()

        for side, preset in (("left", _LEFT_READY_POSE), ("right", _RIGHT_READY_POSE)):
            side_layout = QVBoxLayout()

            side_form = QFormLayout()
            for i in range(1, 8):
                name = f"{side}_fr3_joint{i}"
                edit = QLineEdit()
                edit.setPlaceholderText("deg")
                self._joint_edits[name] = edit
                side_form.addRow(f"joint{i}", edit)
            side_layout.addLayout(side_form)

            preset_btn = QPushButton("Ready Pose")
            preset_btn.clicked.connect(
                lambda _, s=side, p=preset: self._apply_joint_preset(s, p))
            side_layout.addWidget(preset_btn)

            clear_btn = QPushButton("Clear")
            clear_btn.clicked.connect(
                lambda _, s=side: self._clear_joint_values(s))
            side_layout.addWidget(clear_btn)

            side_box = QGroupBox(f"{side.capitalize()} FR3")
            side_box.setLayout(side_layout)
            joints_row.addWidget(side_box)

        layout.addLayout(joints_row)

        # ── Scaling factors ───────────────────────────────────────────────
        scale_form = QFormLayout()

        self._vel_scale = QDoubleSpinBox()
        self._vel_scale.setRange(0.0, 1.0)
        self._vel_scale.setSingleStep(0.05)
        self._vel_scale.setValue(0.1)
        scale_form.addRow("Vel scaling", self._vel_scale)

        self._acc_scale = QDoubleSpinBox()
        self._acc_scale.setRange(0.0, 1.0)
        self._acc_scale.setSingleStep(0.05)
        self._acc_scale.setValue(0.1)
        scale_form.addRow("Acc scaling", self._acc_scale)

        layout.addLayout(scale_form)

        self._move_send_btn = QPushButton("Send Goal")
        self._move_send_btn.setStyleSheet(
            "background-color: #2196F3; color: white; font-weight: bold;")
        self._move_send_btn.clicked.connect(self._on_move_send)
        layout.addWidget(self._move_send_btn)

        self._move_status_label = QLabel("")
        self._move_status_label.setAlignment(Qt.AlignCenter)
        self._move_status_label.setStyleSheet("border: 1px solid #aaa; padding: 4px;")
        layout.addWidget(self._move_status_label)

        return move_box

    # ── Teleoperation slots ───────────────────────────────────────────────

    def _on_send(self) -> None:
        self._node.send_goal(
            mode     = self._mode_combo.currentIndex(),
            left_ee  = self._left_ee.text().strip(),
            right_ee = self._right_ee.text().strip(),
            move_ori = self._move_ori.isChecked(),
            pos_mult = self._pos_mult.value(),
            ori_mult = self._ori_mult.value(),
        )

    def _on_cancel(self) -> None:
        self._node.cancel_goal()

    def _on_status_changed(self, text: str) -> None:
        self._status_label.setText(text)

    def _on_save_count_changed(self, count: int) -> None:
        self._save_count_label.setText(str(count))

    # ── Inference slots ───────────────────────────────────────────────────

    def _on_policy_send(self) -> None:
        self._node.send_policy_goal(
            mode     = self._policy_mode_combo.currentIndex(),
            left_ee  = self._policy_left_ee.text().strip(),
            right_ee = self._policy_right_ee.text().strip(),
        )

    def _on_policy_cancel(self) -> None:
        self._node.cancel_policy_goal()

    def _on_inference_status_changed(self, text: str) -> None:
        self._inference_status_label.setText(text)

    # ── MoveToJoint slots ─────────────────────────────────────────────────

    def _apply_joint_preset(self, side: str, preset: list[float]) -> None:
        for i, val in enumerate(preset, start=1):
            self._joint_edits[f"{side}_fr3_joint{i}"].setText(f"{val*_RAD2DEG:.1f}")

    def _clear_joint_values(self, side: str) -> None:
        for i in range(1, 8):
            self._joint_edits[f"{side}_fr3_joint{i}"].clear()

    def _on_move_send(self) -> None:
        joint_names: list[str]   = []
        target_positions: list[float] = []

        for name, edit in self._joint_edits.items():
            text = edit.text().strip()
            if not text:
                continue
            try:
                target_positions.append(float(text)*_DEG2RAD)
                joint_names.append(name)
            except ValueError:
                self._move_status_label.setText(f"Error: invalid value for {name}.")
                return

        if not joint_names:
            self._move_status_label.setText("Error: no joint values set.")
            return

        self._node.send_move_to_joint_goal(
            joint_names      = joint_names,
            target_positions = target_positions,
            vel_scale        = self._vel_scale.value(),
            acc_scale        = self._acc_scale.value(),
        )

    def _on_move_status_changed(self, text: str) -> None:
        self._move_status_label.setText(text)


# ── Entry point ───────────────────────────────────────────────────────────

def main() -> None:
    rclpy.init()
    signals = _NodeSignals()
    node = FR3HuskyILCommand(signals)

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    app = QApplication(sys.argv)
    gui = CommandPanelGUI(node, signals)
    gui.resize(900, 600)
    gui.show()

    exit_code = app.exec_()

    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    spin_thread.join(timeout=2.0)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
