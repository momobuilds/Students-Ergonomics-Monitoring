import argparse
from HR_Calibration2 import HR_Calibration2
from HR_Measurement import HR_Measurement

def HR_Main(mode, port="COM3", duration=600):
    if mode == "C":

        return HR_Calibration2(
            port=port,
            duration=duration
        )

    elif mode == "M":

        return HR_Measurement(
            port=port,
            duration=duration
        )

    else:

        raise ValueError(
            f"Invalid mode: {mode}"
        )

"""
--------------------
USAGE (CLI)
--------------------

Calibration mode:
    python HR_Main.py -C

Calibration with custom settings:
    python HR_Main.py -C --duration 60 --port COM3

Measurement mode:
    python HR_Main.py -M

Custom port example:
    python HR_Main.py -C --port COM5
"""
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="HR Monitoring System"
    )

    parser.add_argument(
        "-C",
        "--calibration",
        action="store_true",
        help="Run HR calibration mode"
    )

    parser.add_argument(
        "-M",
        "--measurement",
        action="store_true",
        help="Run HR measurement mode"
    )

    parser.add_argument(
        "--port",
        default="COM3",
        help="Serial port of CMS50D device"
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Duration in seconds"
    )

    args = parser.parse_args()

    if args.calibration:

        HR_Main(
            mode="C",
            port=args.port,
            duration=args.duration
        )

    elif args.measurement:

        HR_Main(
            mode="M",
            port=args.port,
            duration=args.duration
        )

    else:
        parser.print_help()