import pandas as pd
import numpy as np
import math

file_name = "text_generation_results.csv"

num_topic = 10



all_df = pd.read_csv(file_name)

trial_num = len(all_df.index)/3
print(trial_num)


def map_value_f(x, mapping_dict):
	return mapping_dict[x]


value_dict = {'Very fluent': 5, 'Fluent': 4, 'Somewhat fluent': 3, 'Not fluent': 2, 'Not fluent at all': 1, np.nan: -1}

all_df['fluency_score'] = all_df['Answer.fluency'].map(lambda x: map_value_f(x, value_dict))
print(all_df.dropna(subset=['Answer.fluency']).groupby('Input.model').fluency_score.agg('mean'))
#print(all_df.dropna(subset=['Answer.fluency']).groupby('Input.model').fluency_score.agg('std'))
print(all_df.dropna(subset=['Answer.fluency']).groupby('Input.model').fluency_score.agg('std')/(math.sqrt(trial_num)))

def get_topic_index(x):
	return '|'.join( [topic.split(':')[0].strip() for topic in x.split('|')])

all_df['selected_topics_idx'] = all_df['Input.selected_topics'].map(get_topic_index)

def compute_accuracy(x):
	gt = x[0].split('|')
	if x[1] is np.nan:
		pred = []
	else:
		pred = str(x[1]).split('|')

	gt_array = np.zeros(num_topic)
	for gt_i in gt:
		gt_array[int(gt_i)] = 1
	pred_array = np.zeros(num_topic)
	for pred_i in pred:
		pred_array[int(pred_i)] = 1
	return 1 - np.sum(np.abs(gt_array - pred_array)) / float(num_topic)

all_df['accuracy'] = all_df[ ['selected_topics_idx','Answer.topic'] ].apply(compute_accuracy,axis=1)
print(all_df.groupby('Input.model').accuracy.agg('mean'))
print(all_df.groupby('Input.model').accuracy.agg('std')/(math.sqrt(trial_num)))
#print(all_df['selected_topics_idx'])

def compute_recall(x):
	gt = x[0].split('|')
	if x[1] is np.nan:
		pred = []
	else:
		pred = str(x[1]).split('|')

	gt_array = np.zeros(num_topic)
	for gt_i in gt:
		gt_array[int(gt_i)] = 1
	pred_array = np.zeros(num_topic)
	for pred_i in pred:
		pred_array[int(pred_i)] = 1
	return np.sum(np.logical_and(gt_array,pred_array)) / np.sum(gt_array)

all_df['recall'] = all_df[ ['selected_topics_idx','Answer.topic'] ].apply(compute_recall,axis=1)
print(all_df.groupby('Input.model').recall.agg('mean'))
print(all_df.groupby('Input.model').recall.agg('std')/(math.sqrt(trial_num)))

def compute_precision(x):
	gt = x[0].split('|')
	if x[1] is np.nan:
		pred = []
	else:
		pred = str(x[1]).split('|')

	gt_array = np.zeros(num_topic)
	for gt_i in gt:
		gt_array[int(gt_i)] = 1
	pred_array = np.zeros(num_topic)
	for pred_i in pred:
		pred_array[int(pred_i)] = 1
	if np.sum(pred_array) > 0:
		return np.sum(np.logical_and(gt_array,pred_array)) / np.sum(pred_array)
	else:
		return np.nan

all_df['precision'] = all_df[ ['selected_topics_idx','Answer.topic'] ].apply(compute_precision,axis=1)
print(all_df.dropna(subset=['precision']).groupby('Input.model').precision.agg('mean'))
print(all_df.dropna(subset=['precision']).groupby('Input.model').precision.agg('std')/(math.sqrt(trial_num)))


def count_num_f(x):
	if 'None' == str(x):
		return 0
	else:
		out = len(str(x).replace('None|','').replace('|None','').split('|'))
	#out = len(x.replace('None','').strip().split('|'))
	#print(x, x.replace('None','').strip(), out)
		return out

all_df['topic_gt_count'] = all_df['selected_topics_idx'].map(count_num_f)
print(np.mean(all_df.topic_gt_count))
