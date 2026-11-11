#!/bin/bash

for i in {1..10}
do
    python /home/wonk/work/exercise/c3h8/tb_md/opt.py
    sleep 5
    timeout 30s python /home/wonk/work/exercise/c3h8/tb_md/md.py
    sleep 5
    python initial_generate.py
    sleep 5
done
