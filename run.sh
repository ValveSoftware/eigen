#!/bin/bash
function run() {
    OLD=0
    NEW=0
    NEWP=0
    EXECS=$1
    SIZE=$2
    RUNS=$3
    for ((i = 0; i < $EXECS; i++)) do
        SEL=$(A=$(shuf -i 0-10 -n 1); echo $(($A % 2)))
        if [ $SEL -eq 0 ]; then
            T_OLD=$(./gto $SIZE $SIZE $SIZE $RUNS)
            T_NEW=$(./gt $SIZE $SIZE $SIZE $RUNS)
            T_NEWP=$(./gtp $SIZE $SIZE $SIZE $RUNS)
        else
            T_NEW=$(./gt $SIZE $SIZE $SIZE $RUNS)
            T_NEWP=$(./gtp $SIZE $SIZE $SIZE $RUNS)
            T_OLD=$(./gto $SIZE $SIZE $SIZE $RUNS)
        fi
        NEW=$NEW+$T_NEW
        OLD=$OLD+$T_OLD
        NEWP=$NEWP+$T_NEWP
    done
    SPEED=$(echo "($OLD) / ($NEW)" | bc -l)
    SPEEDP=$(echo "($OLD) / ($NEWP)" | bc -l)
    echo "$SIZE -> $SPEED $SPEEDP"
}

run $1 16 500
run $1 21 500
run $1 32 500
run $1 53 500
run $1 64 100
run $1 97 100
run $1 128 50
run $1 203 50
run $1 256 10
run $1 673 10
run $1 1024 5
run $1 2048 2
