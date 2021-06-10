# Trains yolact_base_config with a batch_size of 5. For the 550px models, 1 batch takes up around 1.5 gigs of VRAM, so specify accordingly.
# python train_wrapper_yolact.py --config=yolact_plus_resnet50_mp3d_2021_config --batch_size=16 --num_workers=4 --log_folder=/habitat-challenge-data/logs_yolact/ --save_folder=/habitat-challenge-data/weights/ --save_interval=5000 --resume=/habitat-challenge-data/weights/yolact_plus_resnet50_0_10000.pth --start_iter=-1 --validation_epoch=1 --no_log
# python train_wrapper_yolact.py --config=yolact_plus_resnet50_mp3d_2021_config --batch_size=16 --num_workers=4 --log_folder=/habitat-challenge-data/logs_yolact/ --save_folder=/habitat-challenge-data/weights/ --save_interval=2500 --resume=/habitat-challenge-data/weights/yolact_plus_resnet50_0_10000.pth --start_iter=-1 --validation_epoch=1 --no_log


# python train_wrapper_yolact.py --config=yolact_plus_resnet50_mp3d_2021_config --batch_size=16 --num_workers=4 --log_folder=/habitat-challenge-data/logs_yolact/ --save_folder=/habitat-challenge-data/weights/ --save_interval=2500 --resume=/habitat-challenge-data/weights/yolact_plus_resnet50_0_15000.pth --start_iter=-1 --validation_epoch=1 --no_log
# python train_wrapper_yolact.py --config=yolact_plus_resnet50_mp3d_2021_config --batch_size=16 --num_workers=4 --log_folder=/habitat-challenge-data/logs_yolact/ --save_folder=/habitat-challenge-data/weights/ --save_interval=2500 --resume=/habitat-challenge-data/weights/yolact_plus_resnet50_0_25000.pth --start_iter=-1 --validation_epoch=1 --no_log
## get status ##

# python train_wrapper_yolact.py --config=yolact_plus_resnet50_mp3d_2021_config --batch_size=16 --num_workers=4 --log_folder=/habitat-challenge-data/logs_yolact/ --save_folder=/habitat-challenge-data/weights/ --save_interval=2500 --resume=/habitat-challenge-data/weights/yolact_plus_resnet50_0_53200.pth --start_iter=-1 --validation_epoch=1 --no_log
python train_wrapper_yolact.py --config=yolact_plus_resnet50_mp3d_2021_config --batch_size=16 --num_workers=4 --log_folder=/habitat-challenge-data/logs_yolact/ --save_folder=/habitat-challenge-data/weights/ --save_interval=2500 --resume=interrupt --start_iter=-1 --validation_epoch=2 --no_log
status=$?
echo $status

while [ $status -eq 3 ]
do
  python train_wrapper_yolact.py --config=yolact_plus_resnet50_mp3d_2021_config --batch_size=16 --num_workers=4 --log_folder=/habitat-challenge-data/logs_yolact/ --save_folder=/habitat-challenge-data/weights/ --save_interval=2500 --resume=interrupt --start_iter=-1 --validation_epoch=2 --no_log
  status=$?
done

# if [ $? -eq 0 ]
# then
# python train_wrapper_yolact.py --config=yolact_plus_resnet50_mp3d_2021_config --batch_size=16 --num_workers=4 --log_folder=/habitat-challenge-data/logs_yolact/ --save_folder=/habitat-challenge-data/weights/ --save_interval=2500 --resume=interrupt --start_iter=-1 --validation_epoch=1 --no_log
# fi

# python train_wrapper_yolact.py --config=yolact_plus_resnet50_mp3d_2021_config --batch_size=24 --num_workers=4 --log_folder=/habitat-challenge-data/logs_yolact/ --save_folder=/habitat-challenge-data/weights/ --save_interval=2500 --resume=/habitat-challenge-data/weights/yolact_plus_resnet50_0_15000.pth --start_iter=-1 --validation_epoch=1 --no_log
# python train_wrapper_yolact.py --config=yolact_plus_resnet50_mp3d_2021_config --batch_size=16 --num_workers=4 --log_folder=/habitat-challenge-data/logs_yolact/ --save_folder=/habitat-challenge-data/weights/ --save_interval=5000 --resume=interrupt --start_iter=-1 --validation_epoch=1 --no_log
