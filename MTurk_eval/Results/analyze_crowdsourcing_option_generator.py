import pandas as pd
import numpy as np
import math

file_name = "option_generation_results.csv"

all_df = pd.read_csv(file_name)

trial_num = len(all_df.index)/4
print(trial_num)


def map_value_f(x, mapping_dict):
	return mapping_dict[x]


value_dict = {'Very fluent': 5, 'Fluent': 4, 'Somewhat fluent': 3, 'Not fluent': 2, 'Not fluent at all': 1, np.nan: -1}

all_df['fluency_score'] = all_df['Answer.fluency'].map(lambda x: map_value_f(x, value_dict))
print(all_df.dropna(subset=['Answer.fluency']).groupby('Input.topic_mode').fluency_score.agg('mean'))
print(all_df.dropna(subset=['Answer.fluency']).groupby('Input.topic_mode').fluency_score.agg('std')/(math.sqrt(trial_num)))

value_dict = {'Excellent': 5, 'Good': 4, 'Medium': 3, 'Bad': 2, 'Terrible': 1, np.nan: -1}

all_df['overall_score'] = all_df['Answer.sent_overall'].map(lambda x: map_value_f(x, value_dict))
print(all_df.dropna(subset=['Answer.sent_overall']).groupby('Input.topic_mode').overall_score.agg('mean'))
print(all_df.dropna(subset=['Answer.sent_overall']).groupby('Input.topic_mode').overall_score.agg('std')/(math.sqrt(trial_num)))

value_dict = {'Very helpful': 5, 'Helpful': 4, 'Somewhat helpful': 3, 'Not helpful': 2, 'Not helpful at all': 1, np.nan: -1}

all_df['promote_score'] = all_df['Answer.sent_promote'].map(lambda x: map_value_f(x, value_dict))
print(all_df.dropna(subset=['Answer.sent_promote']).groupby('Input.topic_mode').promote_score.agg('mean'))
print(all_df.dropna(subset=['Answer.sent_promote']).groupby('Input.topic_mode').promote_score.agg('std')/(math.sqrt(trial_num)))

num_topic = 10
def ans_intersect(x):
	
	if x[0] is np.nan or x[0] == 'None':
		ans1 = []
	else:
		ans1 = str(x[0]).split('|')

	if x[1] is np.nan or x[1] == 'None':
		ans2 = []
	else:
		ans2 = str(x[1]).split('|')

	ans1_array = np.zeros(num_topic)
	for a_i in ans1:
		if a_i == 'None':
			print ans1
			continue
		ans1_array[int(a_i)] = 1
	ans2_array = np.zeros(num_topic)
	for a_i in ans2:
		if a_i == 'None':
			print ans2
			continue
		ans2_array[int(a_i)] = 1
	return sum( np.logical_or(ans1_array,ans2_array) )

all_df['topic_not_useful'] = all_df[ ['Answer.topic_promote','Answer.topic_likely'] ].apply(ans_intersect,axis=1)
print(10 - all_df.groupby('Input.topic_mode').topic_not_useful.agg('mean'))
print(all_df.groupby('Input.topic_mode').topic_not_useful.agg('std')/(math.sqrt(trial_num)))



#None_value = "None"
def count_num_f(x):
	if 'None' == str(x):
		return 0
	else:
		out = len(str(x).replace('None|','').replace('|None','').split('|'))
	#out = len(x.replace('None','').strip().split('|'))
	#print(x, x.replace('None','').strip(), out)
		return out

#all_df['topic_creativity_count'] = all_df['Answer.topic_creativity'].map(count_num_f)
#print(all_df.groupby('Input.topic_mode').topic_creativity_count.agg('mean'))

#all_df['topic_imagine_count'] = all_df['Answer.topic_imagine'].map(count_num_f)
#print(all_df.groupby('Input.topic_mode').topic_imagine_count.agg('mean'))

all_df['topic_likely_count'] = all_df['Answer.topic_likely'].map(count_num_f)
print(10 - all_df.groupby('Input.topic_mode').topic_likely_count.agg('mean'))
print(all_df.groupby('Input.topic_mode').topic_likely_count.agg('mean')/(math.sqrt(trial_num)))

all_df['topic_promote_count'] = all_df['Answer.topic_promote'].map(count_num_f)
print(10 - all_df.groupby('Input.topic_mode').topic_promote_count.agg('mean'))
print(all_df.groupby('Input.topic_mode').topic_promote_count.agg('std')/(math.sqrt(trial_num)) )

#print(all_df['topic_imagine_count'].to_numpy())
# #with open(file_name) as f_in:

# None_value = 'None'

# Input.topic_mode

# Answer.fluency

# Answer.sent_overall

# Answer.sent_promote

# Answer.topic_creativity	
# Answer.topic_imagine	
# Answer.topic_likely	
# Answer.topic_promote
