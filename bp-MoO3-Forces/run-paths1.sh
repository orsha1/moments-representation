#!/bin/bash
#for j in 32_64_64
#do
#    for i in exp4_coords2
#    do
#        #sed -i 's/features = "CHANGE1"/features = "'"${i}"'"/g' train1.py
#        #sed -i 's/network = "CHANGE2"/network = "'"${j}"'"/g' train1.py
#        nohup /home/sck/shafiro1/py_oenvs/tf/bin/python3 train1.py > ${i}+${j}.out
#        rm ${i}+${j}.out
#        #sed -i 's/features = "'"${i}"'"/features = "CHANGE1"/g' train1.py
#        #sed -i 's/network = "'"${j}"'"/network = "CHANGE2"/g' train1.py
#    done
#done
nohup /home/sck/shafiro1/py_oenvs/tf/bin/python3 train_loop-paths1.py > loop-paths1.out
rm loop-paths1.out


#nohup /home/sck/shafiro1/py_oenvs/tf/bin/python3 train2_2.py > train2_2.out
#rm train2_2.out
#
#nohup /home/sck/shafiro1/py_oenvs/tf/bin/python3 train2_3.py > train2_3.out
#rm train2_3.out