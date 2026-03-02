#!/usr/bin/env python3
import sys
sys.path.insert(0, '../unitree_sdk2/lcm_types')
import lcm
import select

lc = lcm.LCM('udpm://239.255.76.67:7667?ttl=255')
# lc = lcm.LCM('udpm://239.255.76.67:7667?ttl=255?network=192.168.123.164')

print("Listening for LCM messages on udpm://239.255.76.67:7667")
print("Press Ctrl+C to stop\n")
print("-" * 60)

channels_seen = set()

def handler(channel, data):
    if channel not in channels_seen:
        channels_seen.add(channel)
        print(f"\n[NEW CHANNEL] {channel}")
    
    print(f"[{channel}] Received {len(data)} bytes")
    
    try:
        # from transformation_t import transformation_t
        from state_estimator_lcmt import state_estimator_lcmt
        # msg = transformation_t.decode(data)
        # print(f"  Transformation message decoded successfully")
        # if hasattr(msg, 'pos_vicon'):
        #     print(f"    Position: {msg.pos_vicon}")
        # if hasattr(msg, 'quat_vicon'):
        #     print(f"    Orientation: {msg.quat_vicon}")
        msg_state_est = state_estimator_lcmt.decode(data)
        if hasattr(msg_state_est, 'quat'):
            print(f"    State Estimator quat: {msg_state_est.quat}")
        
    except Exception as e:
        pass
        # print(f"  Failed to decode message: {e}")

subscription = lc.subscribe(".*", handler)

try:
    while True:
        timeout = 1.0
        rfds, wfds, efds = select.select([lc.fileno()], [], [], timeout)
        if rfds:
            lc.handle()
except KeyboardInterrupt:
    print("\n" + "-" * 60)
    print(f"\nStopped. Saw {len(channels_seen)} unique channel(s):")
    for ch in sorted(channels_seen):
        print(f"  - {ch}")
