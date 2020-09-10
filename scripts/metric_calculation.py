from metric import  metric_utils
import argparse
import numpy as np

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--prediction_file_path',default= '../translation_out.txt')
arg_parser.add_argument('--true_file_path',default= '../Data/processed_data/test.cmd')
args = arg_parser.parse_args()

predictions =[]
truth =[]

with open(args.prediction_file_path,'r') as pred_file,open(args.true_file_path,'r') as true_file:
    for line in pred_file:
        predictions.append(line)

    for line in true_file:
        truth.append(line)

assert(len(predictions)==len(truth))
score=[]
for predicted_cmd,ground_truth in zip(predictions,truth):
    score.append(metric_utils.compute_metric(predicted_cmd, 1.0 , ground_truth))

score = np.array(score)

print('Average score',score.mean())
print('Variance in score',score.var())