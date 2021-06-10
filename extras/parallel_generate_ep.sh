dt=$(date '+%d/%m/%Y %H:%M:%S')
echo $dt >> /habitat-challenge-data/mine_log_gen0.txt
echo $dt >> /habitat-challenge-data/mine_log_gen1.txt
echo $dt >> /habitat-challenge-data/mine_log_gen2.txt
echo $dt >> /habitat-challenge-data/mine_log_gen3.txt

IDX=0

nohup python extras/generate_ep.py /habitat-challenge-data/scenes_sorted_by_nonzero_instances_of_class_0.json.gz >> /habitat-challenge-data/mine_log_gen0.txt &
res=$!
pids[${IDX}]=$res
IDX=$((IDX + 1))

sleep 15

nohup python extras/generate_ep.py /habitat-challenge-data/scenes_sorted_by_nonzero_instances_of_class_1.json.gz >> /habitat-challenge-data/mine_log_gen1.txt &
res=$!
pids[${IDX}]=$res
IDX=$((IDX + 1))

sleep 15

nohup python extras/generate_ep.py /habitat-challenge-data/scenes_sorted_by_nonzero_instances_of_class_2.json.gz >> /habitat-challenge-data/mine_log_gen2.txt &
res=$!
pids[${IDX}]=$res
IDX=$((IDX + 1))

sleep 15

nohup python extras/generate_ep.py /habitat-challenge-data/scenes_sorted_by_nonzero_instances_of_class_3.json.gz >> /habitat-challenge-data/mine_log_gen3.txt &
res=$!
pids[${IDX}]=$res
IDX=$((IDX + 1))

sleep 15

len=${#pids[@]}
echo ${len}

echo "will wait ${pids[*]}"
for pid in ${pids[*]}; do
    # wait $pid
    tail --pid=$pid -f /dev/null
done
IDX=0
unset pids
