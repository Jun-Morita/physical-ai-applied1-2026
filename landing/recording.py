"""Best-effort encoder shutdown that preserves the episode failure record."""

import subprocess


def finish_video(recorder, log, metadata):
    try:
        if recorder is None:
            return
        try:
            recorder.communicate(b"q\n", timeout=15)
        except (OSError, subprocess.TimeoutExpired) as exc:
            metadata.update(status="incomplete", video_error=str(exc))
            try:
                recorder.kill()
                recorder.communicate(timeout=5)
            except (OSError, subprocess.TimeoutExpired) as stop_exc:
                metadata["video_stop_error"] = str(stop_exc)
        if recorder.returncode != 0:
            metadata.update(status="incomplete")
            metadata.setdefault("video_error", f"ffmpeg exit {recorder.returncode}")
    finally:
        if log:
            try:
                log.close()
            except OSError as exc:
                metadata.update(status="incomplete", video_log_error=str(exc))
