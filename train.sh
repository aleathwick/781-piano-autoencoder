for weights in [1,0.01] [1,0.03] [1,0.1] [1,0.3] [1,1]
do
python run_experiment_seq2seq.py with loss_weights=$weights
done
