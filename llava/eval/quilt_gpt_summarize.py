import json
import os
from collections import defaultdict

import numpy as np

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='ChatGPT-based QA evaluation.')
    parser.add_argument('-d', '--dir', default=None)
    parser.add_argument('-v', '--version', default=None)
    parser.add_argument('-s', '--select', nargs='*', default=None)
    parser.add_argument('-f', '--files', nargs='*', default=['/mnt/f70f6709-366e-49a0-861f-497645d98975/liusn/01output/Quilt-LLaVA-fc_conl/eval/quilt_gpt/review/llava-lm-loss-lora-0605-review.jsonl'])
    parser.add_argument('-c', '--context', default='/mnt/f70f6709-366e-49a0-861f-497645d98975/liusn/00dataset/Quilt_gpt/quilt_gpt_captions.jsonl')
    parser.add_argument('-i', '--ignore', nargs='*', default=[])
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.ignore is not None:
        args.ignore = [int(x) for x in args.ignore]

    if len(args.files) > 0:
        review_files = args.files
    else:
        review_files = [x for x in os.listdir(args.dir) if x.endswith('.jsonl') and (x.startswith('gpt4_text') or x.startswith('reviews_') or x.startswith('review_') or 'review' in args.dir)]
    
    cap = open(os.path.expanduser(args.context))

    for review_file in sorted(review_files):
        config = os.path.basename(review_file).replace('gpt4_text_', '').replace('.jsonl', '')
        if args.select is not None and any(x not in config for x in args.select):
            continue
        if '0613' in config:
            version = '0613'
        else:
            version = '0314'
        if args.version is not None and args.version != version:
            continue
        scores = defaultdict(list)
        print(config)
        with open(os.path.join(args.dir, review_file) if args.dir is not None else review_file) as f:
            for review_str, cap_js in zip(f, cap):
                review = json.loads(review_str)
                cap_str = json.loads(cap_js)
                
                assert review['question_id'] == cap_str['question_id'], 'NOT ALIGNED'
                path_type = cap_str['type']

                if review['question_id'] in args.ignore:
                    continue
                
                if 'category' in review:
                    scores[review['category']].append(review['tuple'])
                    scores['all'].append(review['tuple'])
                else:
                    if 'tuple' in review:
                        scores['all'].append(review['tuple'])
                    else:
                        scores['all'].append(review['score'])

                if path_type:
                    scores[path_type].append(review['tuple'])

        # 每个path_type各有多少samples
        for k, v in sorted(scores.items()):
            print(f'{k} 的样本数为: {len(v)}')

        for k, v in sorted(scores.items()):
            stats = np.asarray(v).mean(0).tolist()
            stats = [round(x, 3) for x in stats]
            # print(k, stats, round(stats[1]/stats[0]*100, 1))
            print(k, round(stats[1]/stats[0]*100, 1), round(stats[0] * 10, 1), round(stats[1] * 10, 1))
        print('=================================')





