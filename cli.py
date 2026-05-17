# cli_run.py
import argparse
from pipeline import run_pipeline_once

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, required=True, help="Input device index")
    ap.add_argument("--label", type=str, default="SOURCE", help="LOCAL or REMOTE")
    ap.add_argument("--seconds", type=int, default=30)
    args = ap.parse_args()

    def on_level(rms, speaking):
        # terminal meter
        bar = "#" * min(50, int(rms * 2000))  # just a rough bar
        print(f"\r{args.label} rms={rms:.5f} speaking={speaking} {bar:<50}", end="")

    def on_text(text):
        print(f"\n[{args.label}] {text}")

    run_pipeline_once(
        device_index=args.device,
        label=args.label,
        seconds=args.seconds,
        on_level=on_level,
        on_text=on_text,
    )

if __name__ == "__main__":
    main()