# the height (empirically) should be smaller than the actual human height, due to inaccuracy of the PICO estimation.
actual_human_height=1.75
python deploy/xrobot_teleop_to_robot_w_hand1.py --robot unitree_g1 \
             --actual_human_height $actual_human_height \
             --target_fps 50 \
             --measure_fps 0 \
             --auto_ground \
            #  --record_dir outputs_bosheng_0204 \
             --disable_draw_targets \
            #  --headless
