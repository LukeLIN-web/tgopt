#!/bin/bash

function run_inference {
    python inference.py -d $1 --model tgat --prefix test --opt-all $2$3 > logs/$1$2.log
}

if [ ! -d "logs/" ]; then
    mkdir logs/
fi

# run_inference jodie-wiki --gpu 0
# run_inference jodie-wiki
# run_inference jodie-mooc --gpu 0
# run_inference jodie-mooc
# run_inference snap-msg --gpu 0
# run_inference snap-msg
# run_inference snap-email --gpu 0
run_inference snap-email
