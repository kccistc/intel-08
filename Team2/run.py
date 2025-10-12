#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Industrial-style relay panel (8 toggles) + 4 sensor indicators + 2 camera previews
- No HTML; pure Python GUI using tkinter
- Integrates with iotdemo.FactoryController (Arduino or FT232)
- Safe to run even without hardware (dummy mode)
- Camera previews for /dev/video0 and /dev/video2 on the right side
"""

import sys
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Optional, Dict

try:
    # Local package layout (as provided by the user)
    from iotdemo import FactoryController, Inputs, Outputs, PyDuino, PyFt232
except Exception as e:
    print("Failed to import iotdemo modules:", e, file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)

import tkinter as tk
from tkinter import ttk

# Optional: OpenCV + PIL for camera preview
_cv2 = None
_ImageTk = None
_Image = None
try:
    import cv2 as _cv2  # type: ignore
    from PIL import Image as _Image
    from PIL import ImageTk as _ImageTk
except Exception as _e:
    print("Camera preview dependencies missing or failed to import:", _e, file=sys.stderr)
    _cv2 = None
    _Image = None
    _ImageTk = None


# --------------------------------------------------------------------------------------
# Hardware Abstraction
# --------------------------------------------------------------------------------------

@dataclass
class RelaySpec:
    name: str
    # When using Arduino: direct pin (Outputs.*)
    arduino_pin: Optional[int] = None
    # FT232 mapping: ('cmd', value_on, value_off) or a callable(controller, on)
    ft232_map: Optional[object] = None


class HW:
    """
    Thin wrapper over FactoryController to present 8 relays as toggles
    and read 4 sensors. Tries to handle both Arduino and FT232.
    """
    def __init__(self, port: Optional[str] = None, debug: bool = True):
        self.ctrl = FactoryController(port=port, debug=debug)
        # Name-mangled internals we may need to poke carefully
        self._device = getattr(self.ctrl, "_FactoryController__device", None)
        self._device_name = getattr(self.ctrl, "_FactoryController__device_name", None)
        self.is_dummy = self.ctrl.is_dummy

        # active-low semantics for Arduino (per FactoryController)
        self.DEV_ON = getattr(FactoryController, "DEV_ON", False)
        self.DEV_OFF = getattr(FactoryController, "DEV_OFF", True)

        # Relay definitions (8 slots)
        self.relays: Dict[int, RelaySpec] = {
            1: RelaySpec("Beacon Red",     arduino_pin=Outputs.BEACON_RED),
            2: RelaySpec("Beacon Orange",  arduino_pin=Outputs.BEACON_ORANGE),
            3: RelaySpec("Beacon Green",   arduino_pin=Outputs.BEACON_GREEN),
            4: RelaySpec("Buzzer",         arduino_pin=Outputs.BEACON_BUZZER),
            5: RelaySpec("LED",            arduino_pin=Outputs.LED),
            6: RelaySpec("Conveyor",       arduino_pin=Outputs.CONVEYOR_EN,
                         ft232_map=("start", 1, 0)),
            7: RelaySpec("Actuator 1",     arduino_pin=Outputs.ACTUATOR_1,
                         ft232_map=("detect", 1, 0)),
            8: RelaySpec("Actuator 2",     arduino_pin=Outputs.ACTUATOR_2,
                         ft232_map=("detect", 2, 0)),
        }

        # Maintain software state for toggles (since some HW may be momentary)
        self.state: Dict[int, bool] = {i: False for i in range(1, 9)}

    # ----------------------------- Relay control helpers ------------------------------

    def _arduino_set(self, pin: int, on: bool):
        if self.is_dummy or self._device is None:
            return

        # Special-case: some kits wire CONVEYOR_EN "active-high" relative to others.
        # User reported "Conveyor runs when OFF" -> invert logic for this pin.
        if pin == Outputs.CONVEYOR_EN:
            # Invert ON/OFF mapping explicitly for conveyor enable
            level = self.DEV_OFF if on else self.DEV_ON
            self._device.set(pin, level)
            # PWM for visibility
            self._device.set(Outputs.CONVEYOR_PWM, 255 if on else 0)
            return

        # Default: active-low
        self._device.set(pin, self.DEV_ON if on else self.DEV_OFF)

        # Conveyor PWM guard (if someone toggles PWM pin directly in future)
        if pin == Outputs.CONVEYOR_EN:
            self._device.set(Outputs.CONVEYOR_PWM, 255 if on else 0)

    def _ft232_set(self, spec: RelaySpec, on: bool):
        if self.is_dummy or self._device is None or spec.ft232_map is None:
            return

        kind, val_on, val_off = spec.ft232_map
        if kind == "start":
            cmd = PyFt232.PKT_CMD_START
            v = PyFt232.PKT_CMD_START_START if on else PyFt232.PKT_CMD_START_STOP
            self._device.set(cmd, v)
            if on:
                for _ in range(3):
                    self._device.set(PyFt232.PKT_CMD_SPEED, PyFt232.PKT_CMD_SPEED_UP)
        elif kind == "detect":
            cmd = PyFt232.PKT_CMD_DETECTION
            v = int(val_on) if on else PyFt232.PKT_CMD_DETECTION_0
            self._device.set(cmd, v)

    def set_relay(self, idx: int, on: bool):
        self.state[idx] = on
        spec = self.relays[idx]

        if self._device_name == "arduino":
            if spec.arduino_pin is not None:
                self._arduino_set(int(spec.arduino_pin), on)
        elif self._device_name == "ft232":
            self._ft232_set(spec, on)

    def get_sensor_states(self):
        """
        Returns dict with 4 sensor booleans:
          - start_button (pressed = True)
          - stop_button (pressed = True)
          - sensor_1 (blocked = True)
          - sensor_2 (blocked = True)
        """
        res = {"start_button": False, "stop_button": False, "sensor_1": False, "sensor_2": False}

        # Arduino path: we can read inputs
        if self._device_name == "arduino" and self._device is not None:
            try:
                res["start_button"] = (self._device.get(Inputs.START_BUTTON) == 0)
                res["stop_button"]  = (self._device.get(Inputs.STOP_BUTTON) == 0)
                res["sensor_1"]     = (self._device.get(Inputs.PHOTOELECTRIC_SENSOR_1) == 0)
                res["sensor_2"]     = (self._device.get(Inputs.PHOTOELECTRIC_SENSOR_2) == 0)
            except Exception:
                pass
        else:
            # Best-effort for FT232/dummy
            try:
                res["sensor_1"] = bool(getattr(self.ctrl, "defect_sensor_status", False))
                res["sensor_2"] = bool(getattr(self.ctrl, "color_sensor_status", False))
            except Exception:
                pass

        return res

    def close(self):
        try:
            self.ctrl.close()
        except Exception:
            pass


# --------------------------------------------------------------------------------------
# GUI
# --------------------------------------------------------------------------------------

class App(tk.Tk):
    def __init__(self, hw: HW):
        super().__init__()
        self.title("Factory Relay Panel")
        self.geometry("1080x640")
        self.minsize(900, 560)

        self.hw = hw

        # Styles
        self.style = ttk.Style(self)
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        # Layout: left control panel, right camera stack
        self.container = ttk.Frame(self, padding=6)
        self.container.pack(fill=tk.BOTH, expand=True)
        self.container.columnconfigure(0, weight=2)  # left
        self.container.columnconfigure(1, weight=3)  # right

        self._build_left_panel(self.container)
        self._build_right_cameras(self.container)

        self._poll_sensors()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ----------------------------- Left Panel (Controls) ------------------------------

    def _build_left_panel(self, parent):
        left = ttk.Frame(parent)
        left.grid(row=0, column=0, sticky="nsew")
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)

        # Top: device status
        top = ttk.Frame(left, padding=6)
        top.grid(row=0, column=0, sticky="ew")
        dev_label = f"Device: {'Dummy' if self.hw.is_dummy else (self.hw._device_name or 'Unknown')}"
        ttk.Label(top, text=dev_label, font=("Arial", 12, "bold")).pack(side=tk.LEFT)

        # Middle: 8 relay toggles (4x2 grid)
        mid = ttk.Frame(left, padding=6)
        mid.grid(row=1, column=0, sticky="nsew")
        self.relay_vars: Dict[int, tk.BooleanVar] = {}
        self.relay_buttons: Dict[int, tk.Button] = {}
        self.relay_lamps: Dict[int, tk.Canvas] = {}

        grid_cfg = [(0,0),(0,1),(0,2),(0,3),
                    (1,0),(1,1),(1,2),(1,3)]

        for idx in range(1, 9):
            frm = ttk.Labelframe(mid, text=f"Relay {idx}: {self.hw.relays[idx].name}", padding=8)
            r, c = grid_cfg[idx-1]
            frm.grid(row=r, column=c, padx=6, pady=6, sticky="nsew")

            var = tk.BooleanVar(value=self.hw.state[idx])
            self.relay_vars[idx] = var

            # Status lamp
            lamp = tk.Canvas(frm, width=26, height=26, highlightthickness=0)
            lamp.grid(row=0, column=0, padx=6, pady=6, sticky="w")
            self.relay_lamps[idx] = lamp

            # Toggle button
            btn = tk.Button(frm, text="OFF", relief=tk.RAISED, font=("Arial", 12, "bold"),
                            command=lambda i=idx: self._toggle(i))
            btn.grid(row=0, column=1, padx=6, pady=6, sticky="ew")
            self.relay_buttons[idx] = btn

            frm.columnconfigure(1, weight=1)
            self._refresh_relay_visual(idx)

        for i in range(2):
            mid.rowconfigure(i, weight=1)
        for j in range(4):
            mid.columnconfigure(j, weight=1)

        # Bottom: sensors + master start/stop
        bottom = ttk.Frame(left, padding=6)
        bottom.grid(row=2, column=0, sticky="ew")

        self.sensor_labels = {
            "start_button": ttk.Label(bottom, text="START: -", width=16, anchor="center"),
            "stop_button": ttk.Label(bottom, text="STOP: -", width=16, anchor="center"),
            "sensor_1": ttk.Label(bottom, text="SENSOR1: -", width=16, anchor="center"),
            "sensor_2": ttk.Label(bottom, text="SENSOR2: -", width=16, anchor="center"),
        }
        for key in ("start_button", "stop_button", "sensor_1", "sensor_2"):
            self.sensor_labels[key].pack(side=tk.LEFT, padx=6)

        ttk.Separator(bottom, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        tk.Button(bottom, text="SYSTEM START", command=self._system_start).pack(side=tk.LEFT, padx=6)
        tk.Button(bottom, text="SYSTEM STOP", command=self._system_stop).pack(side=tk.LEFT, padx=6)

    # ----------------------------- Right Panel (Cameras) ------------------------------

    def _build_right_cameras(self, parent):
        right = ttk.Frame(parent)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        self._cams = []
        if _cv2 is not None and _ImageTk is not None and _Image is not None:
            self._cams = [
                self._make_cam_widget(right, title="Camera 0 (/dev/video0)", device_index=0, row=0),
                self._make_cam_widget(right, title="Camera 2 (/dev/video2)", device_index=2, row=1),
            ]
        else:
            # Fallback text if cv2/PIL not available
            ttk.Label(right, text="Camera preview unavailable (cv2/PIL missing).",
                      anchor="center").grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

    def _make_cam_widget(self, parent, title: str, device_index: int, row: int):
        frame = ttk.Labelframe(parent, text=title, padding=6)
        frame.grid(row=row, column=0, sticky="nsew", padx=6, pady=6)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        label = tk.Label(frame, text="Opening...", anchor="center")
        label.grid(row=0, column=0, sticky="nsew")

        cap = None
        try:
            cap = _cv2.VideoCapture(device_index)
            if not cap.isOpened():
                label.configure(text=f"Cannot open camera index {device_index}")
                cap = None
        except Exception as e:
            label.configure(text=f"Camera {device_index} error: {e}")
            cap = None

        cam = {"cap": cap, "label": label, "last_frame_ts": 0.0}
        # start periodic update
        self.after(50, lambda: self._update_cam_frame(cam, device_index))
        return cam

    def _update_cam_frame(self, cam, device_index: int):
        if _cv2 is None or _ImageTk is None or _Image is None:
            return

        cap = cam["cap"]
        label = cam["label"]
        if cap is None:
            # no device, keep message
            return

        ret, frame = cap.read()
        if not ret or frame is None:
            label.configure(text=f"Camera {device_index}: no frame")
            self.after(200, lambda: self._update_cam_frame(cam, device_index))
            return

        # Convert BGR->RGB, resize to fit label bounds roughly
        rgb = _cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        # Keep a manageable preview size
        max_w, max_h = 640, 300
        scale = min(max_w / float(w), max_h / float(h))
        if scale < 1.0:
            new_size = (int(w * scale), int(h * scale))
            rgb = _cv2.resize(rgb, new_size, interpolation=_cv2.INTER_AREA)

        img = _Image.fromarray(rgb)
        imgtk = _ImageTk.PhotoImage(image=img)
        label.imgtk = imgtk  # keep reference
        label.configure(image=imgtk, text="")

        # schedule next frame
        self.after(50, lambda: self._update_cam_frame(cam, device_index))

    # ----------------------------- UI behavior ----------------------------------

    def _toggle(self, idx: int):
        new_state = not self.relay_vars[idx].get()
        self.relay_vars[idx].set(new_state)
        self.hw.set_relay(idx, new_state)
        self._refresh_relay_visual(idx)

    def _refresh_relay_visual(self, idx: int):
        on = self.relay_vars[idx].get()

        # Lamp color
        lamp = self.relay_lamps[idx]
        lamp.delete("all")
        color = "#2ecc71" if on else "#7f8c8d"  # green / gray
        lamp.create_oval(2, 2, 26, 26, fill=color, outline="black")

        # Button look
        btn = self.relay_buttons[idx]
        btn.configure(text="ON" if on else "OFF")
        btn.configure(relief=tk.SUNKEN if on else tk.RAISED)
        btn.configure(bg="#a3e6b1" if on else "#eeeeee", activebackground="#d5f5e3" if on else "#f2f2f2")

    def _system_start(self):
        try:
            self.hw.ctrl.system_start()
        except Exception:
            traceback.print_exc()

    def _system_stop(self):
        try:
            self.hw.ctrl.system_stop()
        except Exception:
            traceback.print_exc()

    def _poll_sensors(self):
        try:
            st = self.hw.get_sensor_states()
            self._set_sensor_label("start_button", st["start_button"])
            self._set_sensor_label("stop_button",  st["stop_button"])
            self._set_sensor_label("sensor_1",     st["sensor_1"])
            self._set_sensor_label("sensor_2",     st["sensor_2"])
        except Exception:
            traceback.print_exc()

        # poll every 200 ms
        self.after(200, self._poll_sensors)

    def _set_sensor_label(self, key: str, active: bool):
        lbl = self.sensor_labels[key]
        lbl.configure(text=f"{key.upper().replace('_', ' ')}: {'ON' if active else 'OFF'}")
        lbl.configure(background="#f5b7b1" if active else "#d5dbdb")

    def _on_close(self):
        # Release cameras
        try:
            if hasattr(self, "_cams"):
                for cam in self._cams:
                    cap = cam.get("cap")
                    try:
                        if cap is not None:
                            cap.release()
                    except Exception:
                        pass
        except Exception:
            pass

        try:
            self.hw.close()
        except Exception:
            pass
        self.destroy()


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main():
    # Port auto-detection inside FactoryController will try /dev/ttyACM* on Linux.
    hw = HW(port=None, debug=True)
    app = App(hw)
    app.mainloop()


if __name__ == "__main__":
    main()
