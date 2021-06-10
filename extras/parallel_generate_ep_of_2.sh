dt=$(date '+%d/%m/%Y %H:%M:%S')
# echo $dt >> /habitat-challenge-data/mine_log_gen0.txt
echo $dt >> /habitat-challenge-data/mine_log_gen1.txt

IDX=0

# nohup python extras/generate_ep_scene_first.py /habitat-challenge-data/scenes_with_containing_classes_0_of_2.json.gz >> /habitat-challenge-data/mine_log_gen0.txt &
# res=$!
# pids[${IDX}]=$res
# IDX=$((IDX + 1))
#
# sleep 15

# nohup python extras/generate_ep_scene_first.py /habitat-challenge-data/scenes_with_containing_classes_1_of_2.json.gz >> /habitat-challenge-data/mine_log_gen1.txt &
# nohup python extras/generate_ep_scene_first.py /habitat-challenge-data/scenes_with_containing_classes_special.json.gz >> /habitat-challenge-data/mine_log_gen1.txt &
nohup python extras/generate_ep_scene_first.py /habitat-challenge-data/scenes_with_containing_classes_special_scene_with_all_classes.json.gz >> /habitat-challenge-data/mine_log_gen1.txt &
res=$!
pids[${IDX}]=$res
IDX=$((IDX + 1))


len=${#pids[@]}
echo ${len}

echo "will wait ${pids[*]}"
for pid in ${pids[*]}; do
    # wait $pid
    tail --pid=$pid -f /dev/null
done
IDX=0
unset pids
