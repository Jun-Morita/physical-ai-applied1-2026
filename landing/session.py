"""Windows stdlib runner: record, fly, inspect and review FlightGear episodes."""
import argparse
import hashlib
import json
import socket
import subprocess
import time
import urllib.request
from pathlib import Path
from .core import SCHEMA, Policy, action_packet, digest, evaluate, features, packet, read_json


def prop(path):
    with urllib.request.urlopen('http://127.0.0.1:8080/json' + path, timeout=3) as response:
        return json.load(response).get('value')


def set_prop(path, value, kind='bool'):
    parent, name = path.rsplit('/', 1)
    req = urllib.request.Request('http://127.0.0.1:8080/json' + parent,
          data=json.dumps({'name': name, 'type': kind, 'value': value}).encode(),
          headers={'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=3):
        pass


def pause(value):
    for name in ('master', 'clock'):
        set_prop('/sim/freeze/' + name, value)


def rows_at(path):
    return [json.loads(line) for line in (Path(path) / 'samples.jsonl').read_text().splitlines()]


def run(args):
    scenario = read_json(args.scenario)
    policy = Policy(read_json(args.policy)) if args.policy else None
    if policy and policy.model['scenario_hash'] != digest(scenario):
        raise ValueError('Policy and scenario differ; use the training scenario')
    if prop('/sim/aircraft') != scenario['aircraft']:
        raise ValueError('Running aircraft differs from scenario')
    if not prop('/sim/sceneryloaded') or not prop('/sim/signals/fdm-initialized'):
        raise ValueError('Wait until FlightGear scenery and FDM are ready')
    if not prop('/sim/freeze/master') or not prop('/sim/freeze/clock'):
        raise ValueError('Start from a paused scenario')
    if prop('/sim/physical-ai/scenario-hash') != digest(scenario):
        raise ValueError('Restart with start_flightgear.ps1 and this scenario')
    if bool(prop('/sim/input/no-joystick-input')) != bool(policy):
        raise ValueError('Launcher Mode must match human/ai session')
    for name, expected in (('flaps', scenario['controls']['flaps']),):
        if abs(float(prop('/controls/flight/' + name)) - expected) > .02:
            raise ValueError('Configure flaps to the scenario value before starting')
    if not prop('/controls/gear/gear-down-command'):
        raise ValueError('Lower landing gear before starting')
    path = Path(args.output)
    path.mkdir(parents=True, exist_ok=False)
    metadata = {'schema': SCHEMA, 'source': 'ai' if policy else 'human', 'status': 'running',
                'outcome': 'unreviewed', 'scenario': scenario, 'scenario_hash': digest(scenario),
                'policy_hash': digest(policy.model) if policy else None, 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'implementation_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
                'aircraft_lock': read_json(Path(__file__).resolve().parents[1] / 'config/aircraft-lock.json'),
                'video': bool(args.video), 'playback_speed': 1, 'rows': 0}
    def save():
        temp = path / 'metadata.tmp'
        temp.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        temp.replace(path / 'metadata.json')
    save()
    recorder = None
    video_log = None
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rows = []
    try:
        sock.bind(('127.0.0.1', scenario['telemetry_port']))
        sock.settimeout(args.timeout)
        if args.video:
            video_log = (path / 'ffmpeg.log').open('w')
            recorder = subprocess.Popen([args.ffmpeg, '-hide_banner', '-nostats', '-f', 'gdigrab', '-framerate', '20',
                       '-i', 'title=FlightGear', '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
                       '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-pix_fmt', 'yuv420p', str(path / 'flight.mp4')],
                       stdin=subprocess.PIPE, stdout=video_log, stderr=video_log)
            time.sleep(1)
            if recorder.poll() is not None:
                raise RuntimeError('FFmpeg failed; inspect ffmpeg.log and FlightGear window title')
        # Establish a valid telemetry channel before releasing the paused simulator.
        packet(sock.recv(65535))
        pause(False)
        last_time = None
        started = time.monotonic()
        with (path / 'samples.jsonl').open('w', encoding='utf-8') as stream:
            while time.monotonic() - started < args.seconds:
                state = packet(sock.recv(65535))
                # Consume the newest queued packet so AI never steers using an old backlog.
                sock.setblocking(False)
                dropped = 0
                try:
                    while True:
                        state = packet(sock.recv(65535)); dropped += 1
                except BlockingIOError:
                    pass
                finally:
                    sock.settimeout(args.timeout)
                if last_time is not None and state['time'] < last_time:
                    raise RuntimeError('Simulation time reset; start a separate episode')
                if state['paused'] or state['clock_paused']:
                    continue
                if last_time == state['time']:
                    continue
                command = policy.predict(features(state, scenario)) if policy else None
                if command is not None:
                    sender.sendto(action_packet(command, scenario['engine_count']), ('127.0.0.1', scenario['control_port']))
                row = {'state': state, 'command': command, 'wall_elapsed': time.monotonic() - started, 'dropped_packets': dropped}
                stream.write(json.dumps(row) + '\n'); stream.flush()
                rows.append(row)
                last_time = state['time']
                if recorder and recorder.poll() is not None:
                    raise RuntimeError('Video capture stopped unexpectedly')
                if state['crashed']:
                    raise RuntimeError('Simulator reports a crash')
        metadata['status'] = 'complete'
    except KeyboardInterrupt:
        metadata['status'] = 'complete' if rows else 'incomplete'
    except Exception as exc:
        metadata.update(status='incomplete', error=str(exc))
        raise
    finally:
        try:
            pause(True)
        except Exception as exc:
            metadata.update(status='incomplete', pause_error=str(exc))
            print('Could not pause FlightGear:', exc)
        sock.close(); sender.close()
        if recorder:
            try:
                recorder.communicate(b'q\n', timeout=15)
            except subprocess.TimeoutExpired:
                recorder.kill(); recorder.communicate()
                metadata.update(status='incomplete', video_error='capture stop timeout')
            if recorder.returncode != 0:
                metadata.update(status='incomplete', video_error=f'ffmpeg exit {recorder.returncode}')
        if video_log: video_log.close()
        metadata['rows'] = len(rows)
        metadata['evaluation'] = evaluate(rows, scenario)
        save()
        print(json.dumps(metadata, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    for name in ('record', 'fly'):
        q = sub.add_parser(name)
        q.add_argument('--scenario', required=True, type=Path)
        q.add_argument('--output', required=True, type=Path)
        q.add_argument('--seconds', type=float, default=240)
        q.add_argument('--timeout', type=float, default=3)
        q.add_argument('--video', action='store_true')
        q.add_argument('--ffmpeg', default='ffmpeg')
        q.add_argument('--policy', type=Path, required=name == 'fly', default=None)
    for name in ('inspect', 'label'):
        q = sub.add_parser(name)
        q.add_argument('episode', type=Path)
        if name == 'label':
            q.add_argument('--outcome', choices=['success', 'failure', 'exclude'], required=True)
            q.add_argument('--note', required=True)
    a = p.parse_args()
    if a.command in ('record', 'fly'):
        if a.seconds <= 0 or a.timeout <= 0: p.error('Durations must be positive')
        if a.command == 'record' and a.policy: p.error('Use fly for AI control')
        run(a)
    else:
        meta = read_json(a.episode / 'metadata.json')
        meta['evaluation'] = evaluate(rows_at(a.episode), meta['scenario'])
        if a.command == 'label':
            if meta['source'] != 'human' or meta['status'] != 'complete':
                raise ValueError('Only complete human episodes can be reviewed for training')
            meta.update(outcome=a.outcome, review_note=a.note)
            (a.episode / 'metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
        print(json.dumps(meta, indent=2))

if __name__ == '__main__':
    main()
