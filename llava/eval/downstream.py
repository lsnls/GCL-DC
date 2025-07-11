import os
import argparse
import json
import matplotlib.pyplot as plt
import re
import numpy as np
import nltk
# nltk.download('averaged_perceptron_tagger')
# nltk.download('punkt')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')
from sklearn.metrics import recall_score, f1_score, roc_auc_score, precision_score
from sksurv.metrics import concordance_index_censored
import json
from tqdm import tqdm


def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--json_path', default='/home/liusn/02resources/01output/qllava_wsi_output/qllava-lora-0516/WsiVQA_diagnosis_w_ans.json', type=str, help='path to wsi-text pairs')
    args, unparsed = parser.parse_known_args()

    for arg in vars(args):
        if vars(args)[arg] == 'True':
            vars(args)[arg] = True
        elif vars(args)[arg] == 'False':
            vars(args)[arg] = False

    return args

def clean_report_brca(report):
    report_cleaner = lambda t: (t.replace('\n', ' ').replace('  ', ' ') \
        .replace('  ', ' ').replace('  ', ' ')\
        .replace(' 10. ', ' ').replace(' 11. ', ' ').replace(' 12. ', ' ').replace(' 13. ', ' ').replace(' 14.', ' ')    \
        .replace(' 1. ', ' ').replace(' 2. ', ' ') \
        .replace(' 3. ', ' ').replace(' 4. ', ' ').replace(' 5. ', ' ').replace(' 6. ', ' ').replace(' 7. ', ' ').replace(' 8. ', ' ') .replace(' 9. ', ' ')   \
        .replace('A: ', ' ').replace('B: ', ' ').replace('C: ', ' ').replace('D: ', ' ') \
        .strip().lower() + ' ').split('. ')
    sent_cleaner = lambda t: re.sub('[#,?;*!^&_+():-\[\]{}]', '', t.replace('"', '').
                                replace('\\', '').replace("'", '').strip().lower())
    tokens = [sent_cleaner(sent) for sent in report_cleaner(report)]
    report = ' . '.join(tokens)
    return report

def is_idc(text):
    # 导管癌
    if 'ductal carcinoma' in text:
        return 1
    else:
        return 0
def is_pr(text):
    if 'positive' in text:
        return 1
    else:
        return 0

def entity_match(reports, gt):
    reports = clean_report_brca(reports)
    gt = clean_report_brca(gt)
    entities = []
    entities_gt = []

    sentence = nltk.sent_tokenize(reports)
    for sent in sentence:
        for c in nltk.pos_tag(nltk.word_tokenize(sent)):

            if c[1].startswith('NN'):
                if not re.sub('([^\u0061-\u007a])', '', c[0]) == '':
                    entities.append(c[0])

    sentence = nltk.sent_tokenize(gt)
    for sent in sentence:
        for c in nltk.pos_tag(nltk.word_tokenize(sent)):
            if c[1].startswith('NN'):
                if not re.sub('([^\u0061-\u007a])', '', c[0]) == '':
                    entities_gt.append(c[0])

    count = 0
    for e in entities:
        if e in entities_gt:
            count += 1
    if len(entities) == 0:
        return False
    pr = count / len(entities)

    count = 0
    for e in entities_gt:
        if e in entities:
            count += 1
    if len(entities_gt) == 0:
        return False
    rc = count / len(entities_gt)

    f = 2 * rc * pr / (rc + pr + 0.00001)

    return f

def main():
    args = get_args_parser()
    file_name = args.json_path
    subtype_pred = []
    subtype_target = []
    pr_pred = []
    pr_target = []
    all_event_times = []
    all_estimate = []
    result = {}
    fact_vol = 0
    fact_count = 0

    # print(file_name)
    with open(file_name) as f:
        data = json.loads(f.read())
        for item in tqdm(data):
            # brca subtyping
            tgt = clean_report_brca(item['gt'])
            predict = clean_report_brca(item['pred'])
            qs = clean_report_brca(item['question'])
            # fact entity reward
            if True:
                fact = entity_match(predict, tgt)
                if fact:
                    fact_vol += fact
                    fact_count += 1

            if 'logical' in qs:
                if not ('ductal carcinoma' in tgt or 'lobular carcinoma' in tgt):
                    continue

                subtype_pred.append(is_idc(predict))
                subtype_target.append(is_idc(tgt))
            # pr prediction
            if 'receptor' in qs:
                if not tgt in ('negative', 'positive'):
                    continue

                pr_pred.append(is_pr(predict))
                pr_target.append(is_pr(tgt))

            if 'survival time' in qs:
                if not predict.isdecimal():
                    continue
                all_event_times.append(eval(tgt))
                all_estimate.append(eval(predict))
    print(f'len(subtype_pred):{len(subtype_pred)}, len(pr_pred):{len(pr_pred)}, len(all_event_times):{len(all_event_times)}')
    r = recall_score(subtype_pred, subtype_target)
    f1 = f1_score(subtype_pred, subtype_target)
    p = precision_score(subtype_pred, subtype_target)
    # print(f'subtype_pred: {subtype_pred}')
    # print(f'subtype_target: {subtype_target}')

    pr_r = recall_score(pr_pred, pr_target)
    pr_p = precision_score(pr_pred, pr_target)
    pr_f1 = f1_score(pr_pred, pr_target)
    # print(f'pr_pred: {pr_pred}')
    # print(f'pr_target: {pr_target}')
    cindex = concordance_index_censored([True] * len(all_estimate), all_event_times, all_estimate, tied_tol=1e-08)[0]
    result.update({'subtype_r': r, 'subtype_p': p, 'subtype_f1': f1, 'pr_r': pr_r, 'pr_p': pr_p, 'pr_f1': pr_f1,
                   'fact': fact_vol / fact_count})
    print(result)
    print(f'cindex:{cindex}')

if __name__ == '__main__':
    main()
    print('done')