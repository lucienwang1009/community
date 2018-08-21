<<<<<<< HEAD
#/bin/bash 
# runs benchmark and reports time to convergence 
# to use the script: 
=======
#/bin/bash
# runs benchmark and reports time to convergence
# to use the script:
>>>>>>> 15d07f7... 1. add download_dataset.sh and verify_dataset.sh
#   run_and_time.sh <random seed 1-5>


set -e

<<<<<<< HEAD
# start timing 
=======
# start timing
>>>>>>> 15d07f7... 1. add download_dataset.sh and verify_dataset.sh
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"


# run benchmark

seed=${1:-1}

echo "running benchmark with seed $seed"
<<<<<<< HEAD
# Quality of 0.2 is roughly a few hours of work
# 0.749 is the final target quality
./run.sh $seed 0.749
sleep 3 
=======
./run.sh $seed 0.377
sleep 3
>>>>>>> 15d07f7... 1. add download_dataset.sh and verify_dataset.sh
ret_code=$?; if [[ $ret_code != 0 ]]; then exit $ret_code; fi


# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"


<<<<<<< HEAD
# report result 
result=$(( $end - $start )) 
=======
# report result
result=$(( $end - $start ))
>>>>>>> 15d07f7... 1. add download_dataset.sh and verify_dataset.sh
result_name="resnet"


echo "RESULT,$result_name,$seed,$result,$USER,$start_fmt"
