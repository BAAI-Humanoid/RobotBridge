export PYTHONPATH=$(pwd):$PYTHONPATH
python deploy/axisstudio_to_gmr_retarget.py \
 --robot unitree_g1 \
 --target-fps 50 \
 #  --record-dir recordings_0204 \
#  --record-prefix motion_bosheng