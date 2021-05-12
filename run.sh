#!/bin/bash
echo 'Running with master'
T_OLD1=$(./gto)
echo $T_OLD1
echo 'Running current'
T_NEW1=$(./gt)
echo $T_NEW1
echo 'Running with master'
T_OLD2=$(./gto)
echo $T_OLD2
echo 'Running with master'
T_OLD3=$(./gto)
echo $T_OLD3
echo 'Running current'
T_NEW2=$(./gt)
echo $T_NEW2
echo 'Running with master'
T_OLD4=$(./gto)
echo $T_OLD4
echo 'Running current'
T_NEW3=$(./gt)
echo $T_NEW3
echo 'Running current'
T_NEW4=$(./gt)
echo $T_NEW4
echo 'Running with master'
T_OLD5=$(./gto)
echo $T_OLD5
echo 'Running current'
T_NEW5=$(./gt)
echo $T_NEW5
echo "($T_OLD1 + $T_OLD2 + $T_OLD3 + $T_OLD4 + $T_OLD5) / ($T_NEW1 + $T_NEW2 + $T_NEW3 + $T_NEW4 + $T_NEW5)" | bc -l
