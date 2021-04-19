import pandas as pd
import random


show_num = 8

text_file_name = "gen_log/conditional_generation_STSb_SM.csv"
option_file_name = "gen_log/output_STSb_LDA_lkmeans_gkmeans_NSD_example.csv"
output_file = "paper_table"

text_df = pd.read_csv(text_file_name,dtype=str)
option_df = pd.read_csv(option_file_name,dtype=str)
text_df = text_df[text_df['paragraph_previous'].isna()]
option_df = option_df[option_df['paragraph_previous'].isna()]
uniq_sent = [x for x in list(set( pd.merge(option_df,text_df,on='paragraph_last',how='inner')['paragraph_last'].tolist() )) if len(x)<130 ]
random.shuffle(uniq_sent)
selected_sent = uniq_sent[:show_num]
print(selected_sent)

M = 2
def top_words(input_text):
    input_list = input_text.tolist()[0].split(',')
    out_str = ','.join(input_list[:M])
    return out_str[out_str.index(': ')+2:].replace('%','\%')

table_header="\\multicolumn{4}{|c|}{ {\\Large \\Gape[3pt][1pt]{LDA-global} } } & \\multicolumn{4}{|c|}{ {\\Large Kmeans-local} } & \\multicolumn{4}{|c|}{ {\\Large Ours} } \\\\ \\hline"
text_header="\multicolumn{2}{|c|}{Generator} & \multirow{2}{*}{Generated Text}  \\\\ \cline{1-2} \n Option & Text &   \\\\ \hline"
#method_order= [ ['LDA-global','Ours'], ['Kmeans-local','Ours'], ['Ours','PPLM'], ['None','GPT2'], ['Ours','Ours'] ]
output_conditional = []
show_sent_len = 120
with open(output_file,'w') as f_out:
    for sent in selected_sent:
        option_df_sent = option_df[option_df['paragraph_last'] == sent]
        print(option_df_sent)
        input_prompt_out="\\multicolumn{2}{|c|}{ {\\Large \\Gape[3pt][1pt]{Input Prompt}} } &  \\multicolumn{10}{|c|}{ {\\Large "+sent.replace('&','\&')+"} }  \\\\ \\hline"
        f_out.write(input_prompt_out+'\n'+table_header+'\n')
        LDA_df = option_df_sent[option_df_sent['topic_mode']=='LDA_org']
        kmeans_df = option_df_sent[option_df_sent['topic_mode']=='kmeans_cluster']
        our_df = option_df_sent[option_df_sent['topic_mode']=='NSD_topic']
        for i in range(5):
            f_out.write("{}&{}&{}&{}&{}&{}&{}&{}&{}&{}&{}&{} \\\\ \n".format(i+1, top_words(LDA_df['topic_'+str(i)]), i+6, top_words(LDA_df['topic_'+str(i+5)]),i+1, top_words(kmeans_df['topic_'+str(i)]), i+6, top_words(kmeans_df['topic_'+str(i+5)]),i+1, top_words(our_df['topic_'+str(i)]), i+6, top_words(our_df['topic_'+str(i+5)])))
        f_out.write('\\hline \\hline \n')

        text_df_sent = text_df[text_df['paragraph_last'] == sent]
        pplm_df = text_df_sent[text_df_sent['model']=='PPLM 0']
        org_df = text_df_sent[text_df_sent['model']=='Original 0']
        text_prompt_out = "\\multicolumn{2}{|c|}{Input Prompt} &  "+sent.replace('&','\&')+" \\\\ \\hline"
        output_conditional.append(text_prompt_out)
        output_conditional.append(text_header)
        output_conditional.append( 'LDA-global & Ours & ' + LDA_df['sentence'].tolist()[0][:show_sent_len].replace('%','\%').replace('&','\&') + ' \\\\')
        output_conditional.append( 'Kmeans-local & Ours & ' + kmeans_df['sentence'].tolist()[0][:show_sent_len].replace('%','\%').replace('&','\&') + ' \\\\')
        output_conditional.append( 'Ours & PPLM & ' + pplm_df['sentence'].tolist()[0][:show_sent_len].replace('%','\%').replace('&','\&') + ' \\\\')
        output_conditional.append( 'None & GPT2 & ' + org_df['sentence'].tolist()[0][:show_sent_len].replace('%','\%').replace('&','\&') + ' \\\\')
        output_conditional.append( 'Ours & Ours & ' + our_df['sentence'].tolist()[0][:show_sent_len].replace('%','\%').replace('&','\&').replace('$','\$') + ' \\\\ \\hline \\hline')

    f_out.write('\n'+'\n'.join(output_conditional))



    #LDA-global & Ours & text  \\\\
