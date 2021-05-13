#!/bin/bash
function run() {
    OLD=0
    NEW=0
    EXECS=$1
    SIZE=$2
    RUNS=$3
    for ((i = 0; i < $EXECS; i++)) do
        SEL=$(A=$(shuf -i 0-10 -n 1); echo $(($A % 2)))
        if [ $SEL -eq 0 ]; then
            T_OLD=$(./gto $SIZE $RUNS)
            #echo "Master: $T_OLD"
            OLD=$OLD+$T_OLD
            T_NEW=$(./gt $SIZE $RUNS)
            #echo "Current: $T_NEW"
        else
            T_NEW=$(./gt $SIZE $RUNS)
            #echo "Current: $T_NEW"
            T_OLD=$(./gto $SIZE $RUNS)
            #echo "Master: $T_OLD"
            OLD=$OLD+$T_OLD
        fi
        NEW=$NEW+$T_NEW
    done
    SPEED=$(echo "($OLD) / ($NEW)" | bc -l)
    echo "$SIZE -> $SPEED"
}

run $1 16 500
run $1 32 500
run $1 64 500
run $1 128 100
run $1 256 100
