import queue
import threading
import time
import numpy as np
import lcm
from online.camera_reference_data_lcmt import camera_reference_data_lcmt


class LCMPublisher:
    def __init__(self, channel_name):
        self.lc = lcm.LCM('udpm://239.255.76.67:7667?ttl=255')
        self.channel_name = channel_name

    def publish_q(self, data, root_pos=None, root_rot= None, timestamp=None):
        msg = camera_reference_data_lcmt()
        msg.cam_ref_dof_pos = data
        if root_pos is not None:
            msg.root_pos = root_pos
        if root_rot is not None:
            msg.root_rot = root_rot
        if timestamp is not None:
            msg.timestamp = timestamp

        self.lc.publish(self.channel_name, msg.encode())
        #print(f'published qpos to {self.channel_name}')

class SmoothLCMPublisher:
    def __init__(self, channel_name, fps=30):
        self.lc = lcm.LCM('udpm://239.255.76.67:7667?ttl=255')
        self.channel_name = channel_name
        self.fps = fps

        # buffer for saving data
        self.q_buffer = queue.Queue(maxsize=4)
        self.q_timestamps = queue.Queue(maxsize=4)

        self.running = True
        self.publish_thread = threading.Thread(target=self.publish, daemon=True)
        self.publish_thread.start()

    def get_q(self, timestamp_us, method='linear'):
        q_list = list(self.q_buffer.queue)
        timestamps_list = list(self.q_timestamps.queue)

        # need at least two samples to interpolate
        if len(q_list) < 2 or len(timestamps_list) < 2:
            return None

        # ensure equal length and use the most recent aligned pairs
        n = min(len(q_list), len(timestamps_list))
        q_list = q_list[-n:]
        timestamps_list = timestamps_list[-n:]

        # if target time is outside the buffered range, clamp to ends
        if timestamp_us <= timestamps_list[0]:
            return q_list[0]
        if timestamp_us >= timestamps_list[-1]:
            return q_list[-1]

        # find bracketing interval [i, i+1]
        for i in range(n - 1):
            t0 = timestamps_list[i]
            t1 = timestamps_list[i + 1]
            if t0 <= timestamp_us <= t1:
                if method == 'linear':
                    # guard against identical timestamps
                    if t1 == t0:
                        return q_list[i]
                    alpha = (timestamp_us - t0) / float(t1 - t0)
                    q0 = np.asarray(q_list[i], dtype=float)
                    q1 = np.asarray(q_list[i + 1], dtype=float)
                    q_interp = (1.0 - alpha) * q0 + alpha * q1
                    return q_interp.tolist()
                else:
                    # default to step if unknown method
                    return q_list[i]

        # fallback (should not hit due to earlier checks)
        return q_list[-1]

    def publish(self):
        frame_interval_s = 1.0 / float(self.fps)
        # use monotonic clock to schedule stable 30fps output
        while self.running:
            start_time = time.monotonic()
            # ensure a 2-frame delay for smoother interpolation
            current_time_us = int(time.time() * 1e6) - int(2 * 1e6 / float(self.fps))
            q = self.get_q(current_time_us, method='linear')
            if q is None:
                time.sleep(frame_interval_s)
                continue
            msg = camera_reference_data_lcmt()
            msg.cam_ref_dof_pos = q
            self.lc.publish(self.channel_name, msg.encode())
            # sleep to the next frame
            end_time = time.monotonic()
            time_diff = end_time - start_time
            if time_diff < frame_interval_s:
                time.sleep(frame_interval_s - time_diff)



    def push(self, q):
        if self.q_buffer.full():
            self.q_buffer.get_nowait()
            self.q_timestamps.get_nowait()
        self.q_buffer.put_nowait(q)
        self.q_timestamps.put_nowait(int(time.time() * 1e6))

    def stop(self):
        self.running = False
        # give the loop a moment to exit
        try:
            self.publish_thread.join(timeout=1.0)
        except Exception:
            pass