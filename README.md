To replicate the experimental results, first download the data folder ICPSR_22140 and put in same directory, then run the following commands in terminal.

python3.12 -m venv AFEG
source AFEG/bin/activate
pip install --upgrade pip
pip install ipython matplotlib scipy numpy tqdm pandas torch torch-geometric networkx

python3.12 experiment1.py 0
for i in 10 50 100; do python3.12 experiment2.py 0 $i 0.9; done
for i in {0..4}; do python3.12 experiment3.py $i 300 0; done
for i in {0..4}; do for j in {0..9}; do for eps in 0 0.25 0.5 0.75 1; do python3.12 run_noisyQ.py $i 300 0 $eps $j; done; done; done
for i in {0..4}; do python3.12 plot_noisyQ.py $i 300 0 10; done

