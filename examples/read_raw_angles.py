#!/usr/bin/env python

import argparse
import time

from motorbridge_smart_servo import FashionStarServo


DEFAULT_JOINT_IDS = {
    "shoulder_pan": 0,
    "shoulder_lift": 1,
    "elbow_flex": 2,
    "wrist_flex": 3,
    "wrist_yaw": 4,
    "wrist_roll": 5,
    "gripper": 6,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read raw reBot Arm 102 servo angles without LeRobot calibration."
    )
    parser.add_argument("--port", required=True, help="Serial port, e.g. /dev/ttyUSB0")
    parser.add_argument("--baudrate", type=int, default=1000000)
    parser.add_argument(
        "--interval", type=float, default=0.2, help="Polling interval in seconds"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with FashionStarServo(args.port, baudrate=args.baudrate) as bus:
        # Initialize servos: ping, unlock, and reset the multi-turn counter.
        for servo_id in DEFAULT_JOINT_IDS.values():
            if not bus.ping(servo_id):
                print(f"WARNING: servo id={servo_id} did not respond to ping")
            bus.unlock(servo_id)
            time.sleep(0.01)
            bus.reset_multi_turn(servo_id)

        print("Reading raw servo angles. Press Ctrl+C to stop.")
        try:
            while True:
                values = []
                for joint_name, servo_id in DEFAULT_JOINT_IDS.items():
                    raw_angle = bus.read_raw_angle(servo_id, multi_turn=True)
                    values.append(f"{joint_name}={raw_angle:8.2f}")
                print("  ".join(values))
                time.sleep(args.interval)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
