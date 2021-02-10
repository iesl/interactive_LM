import csv
import random
import getopt
import sys

help_msg = '-p <input_prefix> -o <output_path> -m <method_list_str>'

try:
    opts, args = getopt.getopt(sys.argv[1:], "p:o:m:")
except getopt.GetoptError:
    print(help_msg)
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print(help_msg)
        sys.exit()
    elif opt in ("-p"):
        input_prefix = arg
    elif opt in ("-o"):
        output_path = arg
    elif opt in ("-m"):
        method_list_str = arg

method_list = method_list_str.split('|')
print(method_list_str)

#input_dir = './gen_log'
#input_prefix = 'output_'
#input_prefix = 'output_STS_'
#input_suffix = '_task23_1b.csv'
#input_suffix = '_task23_1b_long_example.csv'
input_suffix = '.csv'
#method_list = ['kmeans_global', 'kmeans_local', 'LDA', 'NSD', 'sample_local']
#method_list = ['kmeans_global', 'kmeans_local', 'NSD']
#method_list = ['LDA_org', 'kmeans_global', 'kmeans_local', 'NSD_3w']
#output_path = './gen_log/output_task23_1b.csv'
#output_path = './gen_log/output_STS_task23_1b.csv'
#output_path = './gen_log/output_STS_task23_1b_long.csv'
#output_path = './gen_log/output_STSb_task23_1b_LDA_gkmeans_3w_long_example.csv'
context_d2_method_d2_line = {}
method_d2_UTF8_count = {}

for method in method_list:
    #file_name = input_dir+'/'+input_prefix+method+input_suffix
    file_name = input_prefix+method+input_suffix
    print(file_name)
    with open(file_name, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        header =  next(spamreader,None)
        for row in spamreader:
            method = row[-2]
            generated_sent = row[-3]
            #method = row[-1]
            #generated_sent = row[-2]
            generated_sent = generated_sent.replace('â',"'").replace('â','-').replace('\n'," ")
            try:
                generated_sent.encode('ascii', 'strict')
            except:
                count = method_d2_UTF8_count.get(method,0)
                method_d2_UTF8_count[method] = count + 1
                continue
            if generated_sent.count(" ") > len(generated_sent) * 0.3:
                count = method_d2_UTF8_count.get(method,0)
                method_d2_UTF8_count[method] = count + 1
                continue
            context = row[0] + ' ' + row[1]
            #row[-2] = generated_sent
            row[-3] = generated_sent
            method_d2_line = context_d2_method_d2_line.get(context, {})
            csv_lines = method_d2_line.get(method,[])
            csv_lines.append(row)
            method_d2_line[method] = csv_lines
            context_d2_method_d2_line[context] = method_d2_line

with open(output_path, 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',quotechar='"')
    spamwriter.writerow(header)
    for context in context_d2_method_d2_line:
        method_d2_line = context_d2_method_d2_line[context]
        min_count = 0
        if len(method_d2_line) == len(method_list):
            min_count = min(map(len,method_d2_line.values()))
        for method in method_d2_line:
            csv_lines = method_d2_line[method]
            for row in random.sample(csv_lines,min_count):
                spamwriter.writerow(row)


print(method_d2_UTF8_count)
