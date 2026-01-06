# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import tkinter as tk
from tkinter import ttk
import threading
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool


class SimControlGUINode(Node):
    """ROS2 node for publishing control messages"""

    def __init__(self, node_name="sim_control_gui_node"):
        super().__init__(node_name)

        # Create publishers
        self.pub_instruction = self.create_publisher(String, "/sim/instruction", 1)
        self.pub_reset = self.create_publisher(Bool, "/sim/reset", 1)
        self.pub_infer_start = self.create_publisher(Bool, "/sim/infer_start", 1)
        self.pub_shuffle = self.create_publisher(Bool, "/sim/shuffle", 1)

        self.get_logger().info("SimControlGUI ROS node initialized")

    def publish_instruction(self, instruction_text: str):
        """Publish instruction message"""
        msg = String()
        msg.data = instruction_text
        self.pub_instruction.publish(msg)
        self.get_logger().info(f"Published instruction: {instruction_text}")

    def publish_reset(self, reset_value: bool = True):
        """Publish reset message"""
        msg = Bool()
        msg.data = reset_value
        self.pub_reset.publish(msg)
        self.get_logger().info(f"Published reset: {reset_value}")

    def publish_infer_start(self, start_value: bool = True):
        """Publish infer_start message"""
        msg = Bool()
        msg.data = start_value
        self.pub_infer_start.publish(msg)
        self.get_logger().info(f"Published infer_start: {start_value}")

    def publish_shuffle(self, shuffle_value: bool = True):
        """Publish shuffle message"""
        msg = Bool()
        msg.data = shuffle_value
        self.pub_shuffle.publish(msg)
        self.get_logger().info(f"Published shuffle: {shuffle_value}")


class SimControlGUI:
    """Simulation Control GUI Plugin"""

    def __init__(self, node_name="sim_control_gui_node"):
        """
        Initialize GUI interface

        Args:
            node_name: ROS2 node name
        """
        self.node_name = node_name
        self.ros_node = None
        self.ros_thread = None
        self.root = None

        # Button press state tracking
        self.start_pressed = False
        self.reset_pressed = False
        self.shuffle_pressed = False
        # Current state of infer_start (controlled by start and reset buttons)
        self.infer_start_state = False

        # Initialize ROS2
        if not rclpy.ok():
            rclpy.init()

        # Create ROS node
        self.ros_node = SimControlGUINode(node_name)

        # Run ROS2 in a separate thread
        self.ros_thread = threading.Thread(target=self._ros_spin, daemon=True)
        self.ros_thread.start()

        # Create GUI
        self._create_gui()

    def _ros_spin(self):
        """Run ROS2 spin in a separate thread"""
        try:
            while rclpy.ok() and self.ros_node is not None:
                rclpy.spin_once(self.ros_node, timeout_sec=0.1)
        except Exception as e:
            if self.ros_node:
                self.ros_node.get_logger().error(f"ROS spin error: {e}")

    def _create_gui(self):
        """Create GUI interface"""
        self.root = tk.Tk()
        self.root.title("Simulation Control Panel")
        self.root.geometry("550x450")
        self.root.minsize(500, 400)  # Set minimum size
        self.root.resizable(True, True)  # Allow resizing

        # Set window background color
        self.root.configure(bg="#f0f0f0")

        # Set window close event
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

        # Create main container
        main_container = tk.Frame(self.root, bg="#f0f0f0", padx=20, pady=20)
        main_container.pack(fill=tk.BOTH, expand=True)

        # Configure grid weights for root window to allow content to expand
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Title area
        title_frame = tk.Frame(main_container, bg="#2c3e50", pady=12)
        title_frame.pack(fill=tk.X, pady=(0, 15))

        title_label = tk.Label(
            title_frame,
            text="🚀 Simulation Control Panel",
            font=("Arial", 16, "bold"),
            bg="#2c3e50",
            fg="white",
        )
        title_label.pack()

        # Instruction input area
        instruction_frame = tk.LabelFrame(
            main_container,
            text="📝 Instruction Input",
            font=("Arial", 10, "bold"),
            bg="#f0f0f0",
            fg="#2c3e50",
            padx=12,
            pady=12,
            relief=tk.RAISED,
            borderwidth=2,
        )
        instruction_frame.pack(fill=tk.X, pady=(0, 12))

        # Input box and button container
        input_container = tk.Frame(instruction_frame, bg="#f0f0f0")
        input_container.pack(fill=tk.X)

        self.instruction_entry = tk.Entry(
            input_container,
            font=("Arial", 11),
            relief=tk.SOLID,
            borderwidth=2,
            highlightthickness=2,
            highlightbackground="#3498db",
            highlightcolor="#2980b9",
        )
        self.instruction_entry.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        self.instruction_entry.bind("<Return>", lambda e: self._on_send_instruction())
        self.instruction_entry.bind("<FocusIn>", lambda e: self.instruction_entry.config(highlightbackground="#2980b9"))

        self.send_btn = tk.Button(
            input_container,
            text="Send",
            command=self._on_send_instruction,
            font=("Arial", 10, "bold"),
            bg="#3498db",
            fg="white",
            activebackground="#2980b9",
            activeforeground="white",
            relief=tk.FLAT,
            padx=20,
            pady=8,
            cursor="hand2",
        )
        self.send_btn.pack(side=tk.RIGHT)

        # Control button area
        control_frame = tk.LabelFrame(
            main_container,
            text="⚙️ Control Operations",
            font=("Arial", 10, "bold"),
            bg="#f0f0f0",
            fg="#2c3e50",
            padx=12,
            pady=12,
            relief=tk.RAISED,
            borderwidth=2,
        )
        control_frame.pack(fill=tk.X, pady=(0, 12))

        # Button container
        button_container = tk.Frame(control_frame, bg="#f0f0f0")
        button_container.pack(fill=tk.X)

        # Start Button - Green Theme
        start_btn = tk.Button(
            button_container,
            text="▶ START",
            font=("Arial", 11, "bold"),
            bg="#27ae60",
            fg="white",
            activebackground="#229954",
            activeforeground="white",
            relief=tk.FLAT,
            padx=25,
            pady=12,
            cursor="hand2",
            borderwidth=0,
        )
        # Bind press and release events (using Button-1 to ensure only left-click response)
        start_btn.bind("<ButtonPress-1>", lambda e: self._on_start_press())
        start_btn.bind("<ButtonRelease-1>", lambda e: self._on_start_release())
        # If mouse leaves button area and button is pressed, also send false
        start_btn.bind("<Leave>", lambda e: self._on_start_release() if self.start_pressed else None)
        start_btn.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        # Reset Button - Orange Theme
        reset_btn = tk.Button(
            button_container,
            text="🔄 RESET",
            font=("Arial", 11, "bold"),
            bg="#e67e22",
            fg="white",
            activebackground="#d35400",
            activeforeground="white",
            relief=tk.FLAT,
            padx=25,
            pady=12,
            cursor="hand2",
            borderwidth=0,
        )
        # Bind press and release events (using Button-1 to ensure only left-click response)
        reset_btn.bind("<ButtonPress-1>", lambda e: self._on_reset_press())
        reset_btn.bind("<ButtonRelease-1>", lambda e: self._on_reset_release())
        # If mouse leaves button area and button is pressed, also send false
        reset_btn.bind("<Leave>", lambda e: self._on_reset_release() if self.reset_pressed else None)
        reset_btn.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 8))

        # Shuffle Button - Purple Theme
        shuffle_btn = tk.Button(
            button_container,
            text="🔀 SHUFFLE",
            font=("Arial", 11, "bold"),
            bg="#9b59b6",
            fg="white",
            activebackground="#8e44ad",
            activeforeground="white",
            relief=tk.FLAT,
            padx=25,
            pady=12,
            cursor="hand2",
            borderwidth=0,
        )
        # Bind press and release events (using Button-1 to ensure only left-click response)
        shuffle_btn.bind("<ButtonPress-1>", lambda e: self._on_shuffle_press())
        shuffle_btn.bind("<ButtonRelease-1>", lambda e: self._on_shuffle_release())
        # If mouse leaves button area and button is pressed, also send false
        shuffle_btn.bind("<Leave>", lambda e: self._on_shuffle_release() if self.shuffle_pressed else None)
        shuffle_btn.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(0, 0))

        # Status display area
        status_frame = tk.LabelFrame(
            main_container,
            text="📊 Status Information",
            font=("Arial", 10, "bold"),
            bg="#f0f0f0",
            fg="#2c3e50",
            padx=12,
            pady=12,
            relief=tk.RAISED,
            borderwidth=2,
        )
        status_frame.pack(fill=tk.BOTH, expand=True)  # Allow status area to expand

        # Status label container
        status_container = tk.Frame(status_frame, bg="#ecf0f1", relief=tk.SUNKEN, borderwidth=1)
        status_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.status_label = tk.Label(
            status_container,
            text="✓ Ready",
            font=("Arial", 10),
            bg="#ecf0f1",
            fg="#27ae60",
            anchor=tk.W,
            padx=10,
            pady=8,
            wraplength=450,  # Allow text wrapping
            justify=tk.LEFT,
        )
        self.status_label.pack(fill=tk.BOTH, expand=True)

    def _on_send_instruction(self):
        """Callback for send instruction button"""
        instruction_text = self.instruction_entry.get().strip()
        if instruction_text:
            if self.ros_node:
                self.ros_node.publish_instruction(instruction_text)
                self.status_label.config(text=f"✓ Instruction sent: {instruction_text}", bg="#d5e8f7", fg="#2980b9")
                # Keep instruction, disable input box (greyed out, not editable)
                self.instruction_entry.config(
                    state="disabled", disabledbackground="#e0e0e0", disabledforeground="#808080"
                )
                # Also disable send button
                self.send_btn.config(state="disabled", bg="#bdc3c7", cursor="arrow")
            else:
                self.status_label.config(text="✗ ROS node not initialized", bg="#fadbd8", fg="#c0392b")
        else:
            self.status_label.config(text="⚠ Please enter instruction", bg="#fef5e7", fg="#f39c12")

    def _on_start_press(self):
        """When Start button is pressed, infer_start sends true and remains true"""
        self.start_pressed = True
        if self.ros_node:
            self.infer_start_state = True
            self.ros_node.publish_infer_start(True)
            self.status_label.config(text="▶ START: infer_start = true", bg="#d5f4e6", fg="#27ae60")
        else:
            self.status_label.config(text="✗ ROS node not initialized", bg="#fadbd8", fg="#c0392b")

    def _on_start_release(self):
        """When Start button is released, infer_start remains true"""
        if self.start_pressed:
            self.start_pressed = False
            # infer_start remains true, does not send false
            if self.ros_node:
                self.status_label.config(text="▶ START: Released, infer_start remains true", bg="#d5f4e6", fg="#27ae60")
            else:
                self.status_label.config(text="✗ ROS node not initialized", bg="#fadbd8", fg="#c0392b")

    def _on_reset_press(self):
        """When Reset button is pressed, reset sends true, infer_start sends false and remains false"""
        self.reset_pressed = True
        if self.ros_node:
            # reset sends true
            self.ros_node.publish_reset(True)
            # infer_start sends false and remains false
            self.infer_start_state = False
            self.ros_node.publish_infer_start(False)
            # Restore input box to editable state (without clearing content)
            self.instruction_entry.config(state="normal")
            # Restore send button to enabled state
            self.send_btn.config(state="normal", bg="#3498db", cursor="hand2")
            self.status_label.config(text="🔄 RESET: ON, infer_start = false", bg="#fdebd0", fg="#e67e22")
        else:
            self.status_label.config(text="✗ ROS node not initialized", bg="#fadbd8", fg="#c0392b")

    def _on_reset_release(self):
        """When Reset button is released, reset sends false, infer_start remains false"""
        if self.reset_pressed:
            self.reset_pressed = False
            if self.ros_node:
                # reset sends false
                self.ros_node.publish_reset(False)
                # infer_start remains false, unchanged
                self.status_label.config(
                    text="🔄 RESET: Released, infer_start remains false", bg="#ecf0f1", fg="#7f8c8d"
                )
            else:
                self.status_label.config(text="✗ ROS node not initialized", bg="#fadbd8", fg="#c0392b")

    def _on_shuffle_press(self):
        """When Shuffle button is pressed, send shuffle = true"""
        self.shuffle_pressed = True
        if self.ros_node:
            self.ros_node.publish_shuffle(True)
            self.status_label.config(text="🔀 SHUFFLE: ON", bg="#e8daef", fg="#9b59b6")
        else:
            self.status_label.config(text="✗ ROS node not initialized", bg="#fadbd8", fg="#c0392b")

    def _on_shuffle_release(self):
        """When Shuffle button is released, send shuffle = false"""
        if self.shuffle_pressed:
            self.shuffle_pressed = False
            if self.ros_node:
                self.ros_node.publish_shuffle(False)
                self.status_label.config(text="🔀 SHUFFLE: Released", bg="#ecf0f1", fg="#7f8c8d")
            else:
                self.status_label.config(text="✗ ROS node not initialized", bg="#fadbd8", fg="#c0392b")

    def _on_closing(self):
        """Window close event handler"""
        if self.ros_node:
            self.ros_node.get_logger().info("Closing GUI...")
        self.destroy()

    def run(self):
        """Run GUI main loop"""
        if self.root:
            self.root.mainloop()

    def destroy(self):
        """Destroy GUI and ROS node"""
        if self.root:
            self.root.destroy()
        if self.ros_node:
            self.ros_node.destroy_node()


def main():
    """Main function for direct script execution"""
    gui = SimControlGUI()
    try:
        gui.run()
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        gui.destroy()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
