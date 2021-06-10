#python eval.py --trained_model=weights/yolact_resnet50_54_800000.pth --score_threshold=0.15 --top_k=15 --image=/home/dvruiz/Pictures/giraffes.jpg

#python eval.py --trained_model=weights/yolact_resnet50_54_800000.pth --score_threshold=0.15 --top_k=15 --video_multiframe=30 --video=/home/dvruiz/pos/codeForFinalThesis/vribot/src/gibson2-ros/examples/ros/gibson2-ros/video/rgb_video_stream.mp4

source ~/externaldrive/pos/catkin_ws_python3/devel/setup.bash

python eval_ros.py --trained_model=weights/yolact_resnet50_54_800000.pth --score_threshold=0.15 --top_k=15 --video_multiframe=1 --video=/home/dvruiz/pos/codeForFinalThesis/vribot/src/gibson2-ros/examples/ros/gibson2-ros/video/rgb_video_stream.mp4
