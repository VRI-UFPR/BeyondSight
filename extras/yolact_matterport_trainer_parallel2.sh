dt=$(date '+%d/%m/%Y %H:%M:%S')
echo $dt >> /habitat-challenge-data/mine_log_gen0.txt
echo $dt >> /habitat-challenge-data/mine_log_gen1.txt
echo $dt >> /habitat-challenge-data/mine_log_gen2.txt
echo $dt >> /habitat-challenge-data/mine_log_gen3.txt
echo $dt >> /habitat-challenge-data/mine_log_gen4.txt

IDX=0

nohup python beyond_agent/grab_viewpoints.py --split 1 >> /habitat-challenge-data/mine_log_gen0.txt &
res=$!
pids[${IDX}]=$res
IDX=$((IDX + 1))

sleep 15

nohup python beyond_agent/grab_viewpoints.py --split 2 >> /habitat-challenge-data/mine_log_gen1.txt &
res=$!
pids[${IDX}]=$res
IDX=$((IDX + 1))

sleep 15

nohup python beyond_agent/grab_viewpoints.py --split 3 >> /habitat-challenge-data/mine_log_gen2.txt &
res=$!
pids[${IDX}]=$res
IDX=$((IDX + 1))

sleep 15

nohup python beyond_agent/grab_viewpoints.py --split 4 >> /habitat-challenge-data/mine_log_gen3.txt &
res=$!
pids[${IDX}]=$res
IDX=$((IDX + 1))

sleep 15

nohup python beyond_agent/grab_viewpoints.py --split 5 >> /habitat-challenge-data/mine_log_gen4.txt &
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
