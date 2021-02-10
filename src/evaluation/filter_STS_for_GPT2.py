from spacy.lang.en import English

#input_path = "/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/dataset_testing/STS/sts_all_years_test"
#output_path = "/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/dataset_testing/STS/sts_all_years_test_longer"
#input_path = "/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/dataset_testing/STS/stsbenchmark/sts-train.csv"
#output_path = "/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/dataset_testing/STS/stsbenchmark/sts-train_longer.csv"
input_path = "./data/raw/stsbenchmark/sts-train.csv"
output_path = "./data/raw/stsbenchmark/sts-train_longer.csv"
stop_word_list = "./resources/stop_word_list"
n_basis = 10

nlp = English()

with open(stop_word_list) as f_in:
    stop_word_org_set = set()
    for line in f_in:
        w = line.rstrip()
        stop_word_org_set.add(w)

sent_type_list = []

with open(input_path) as f_in:
    for line in f_in:
        fields = line.rstrip().split('\t')
        sent_type_list.append( (fields[-2],fields[1]) )
        sent_type_list.append( (fields[-1],fields[1]) )

with open(output_path, 'w') as f_out:
    for sent, type_name in sent_type_list:
        tokens = nlp.tokenizer(sent)
        word_count = 0
        for tok in tokens:
            w = tok.text
            if w not in stop_word_org_set:
                word_count += 1
        if word_count > n_basis:
            f_out.write(sent +'\t'+ type_name +'\n')
