import argparse
import difflib

import torch
import os
import json
from tqdm import tqdm
import shortuuid

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from quilt_utils import *

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    assert 'lora' in model_name.lower() and 'llava' in model_name.lower(), f"Model name {model_name} is not valid. Please check the model path."
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, is_wsi=True)

    # questions = json.load(open(os.path.expanduser(args.question_file), "r")) # read json format questions
    # 读取jsonl格式的questions
    questions = [json.loads(line) for line in open(os.path.expanduser(args.question_file), "r") if line.startswith('{')]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    exist_question = None
    if os.path.exists(answers_file) and os.path.getsize(answers_file) > 0:
        print('already exists!')
        # 截断json文件的最后一行
        with open(answers_file, 'rb+') as f:
            # 将文件指针移动到倒数第二个字节
            f.seek(-2, os.SEEK_END)
            # 向前移动, 直到换行符or文件开头
            while f.read(1) != b'\n' and f.tell() > 0:
                f.seek(-2, os.SEEK_CUR)
            if f.tell() > 0:
                f.seek(f.tell())
                f.truncate()
                print("已删除最后一行。")

        ans_file = open(answers_file, "a")
        with open(answers_file, 'r') as f:
            exist_data = [json.loads(line) for line in f if line.startswith('{')]
        exist_question = [item['question_id'] for item in exist_data]
        print(f'已经有{len(exist_question)}条数据存在了')
    else:
        print('not exists!')
        ans_file = open(answers_file, "w")

    for i, line in enumerate(tqdm(questions)):
        idx = line["id"]

        if exist_question is not None:
            if idx in exist_question:
                continue
        answer_type = line["answer_type"]
        question = line['conversations'][0]
        qs = question['value'].replace('<image>', '').strip()
        cur_prompt = qs

        if 'image' in line:
            image_file = line["image"]
            if image_file.endswith('.pt'):
                image_tensor = torch.load(os.path.join(args.image_folder, image_file))
            else:
                image = Image.open(os.path.join(args.image_folder, image_file))
                image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            images = image_tensor.unsqueeze(0).half().cuda()
            # print(f'images.shape = {images.shape}')

            if getattr(model.config, 'mm_use_im_start_end', False):
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            cur_prompt = '<image>' + '\n' + cur_prompt
        else:
            images = None

        if args.single_pred_prompt:
            # 影响prompt的是qs
            # qs = qs + '\n' + "Answer with the option's letter from the given choices directly."
            # cur_prompt = cur_prompt + '\n' + "Answer with the option's letter from the given choices directly."
            if answer_type == 'CLOSED':
                qs = qs + '\n' + "Answer with the correct option's letter first, followed by the content of the option."
                cur_prompt = cur_prompt + '\n' + "Answer with the correct option's letter first, followed by the content of the option."
            # oepn set不做处理
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        print(f'prompt = {prompt}'); exit()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = [KeywordsStoppingCriteria(keywords, tokenizer, input_ids)] if conv.version == "v0" else None

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=stopping_criteria,
                is_eval=True,
            )
        input_token_len = input_ids.shape[1]
        # print(f'input_ids.shape = {input_ids.shape}. output_ids.shape = {output_ids.shape}, output_ids[:, input_token_len:].shape = {output_ids[:, input_token_len:].shape}')
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        print(f'outputs = {outputs}')
        if outputs.endswith(stop_str):
            # print(f'stop_str = {stop_str}')
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        # prompt for answer
        if args.answer_prompter:
            outputs_reasoning = outputs
            input_ids = tokenizer_image_token(prompt + outputs_reasoning + ' ###\nANSWER:', tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    max_new_tokens=64,
                    use_cache=True,
                    stopping_criteria=stopping_criteria) # [stopping_criteria]

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()
            outputs = outputs_reasoning + '\n The answer is ' + outputs
            print(f'outputs2 = {outputs}')

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "answer_type": answer_type,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

def load_jsonl(f_path):
    data = []
    with open(f_path, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            data.append(item)
    return data

def compute_scores(gts, res):
    """
    :param gts: Dictionary with the image ids and their gold captions,
    :param res: Dictionary with the image ids ant their generated captions
    :print: Evaluation score (the mean of the scores of all the instances) for each measure
    """

    # Set up scorers
    scorers = [
        (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L")
    ]
    eval_res = {}
    # print(f'gts[0] = {gts[0]}, res[0] = {res[0]}')
    # Compute score for each metric
    for scorer, method in scorers:
        try:
            score, scores = scorer.compute_score(gts, res)
        except TypeError:
            score, scores = scorer.compute_score(gts, res)
        if type(method) == list:
            for sc, m in zip(score, method):
                eval_res[m] = sc
        else:
            eval_res[method] = score
    return eval_res

def compute_acc(choices, gts, preds):
    wrong, total = 0, 0
    for k in gts.keys():
        choice = choices[k]
        gt = gts[k][0]
        pred = preds[k][0]
        score = difflib.SequenceMatcher(None, pred, gt).quick_ratio()
        for c in choice:
            tmp = difflib.SequenceMatcher(None, c, gt).quick_ratio()
            if tmp > score:
                wrong += 1
                break
        total += 1
    acc_res = (total - wrong) / total
    return acc_res
def evaluate_metric(args):
    gt = load_jsonl(args.gt)
    pred = load_jsonl(args.pred)

    gt_ids = sorted([item['id'] for item in gt])
    pred_ids = sorted([item['question_id'] for item in pred])
    assert gt_ids == pred_ids, "please make sure pred and gt are exactly matched"

    gts_closed = {item['id']: [normalize_word(item['conversations'][1]['value'])] for item in gt if item['answer_type'] == 'CLOSED'}
    preds_closed = {item['question_id']: [normalize_word(item['text'])] for item in pred if item['answer_type'] == 'CLOSED'}
    choices_closed = {item['id']: item['choices'] for item in gt if item['answer_type'] == 'CLOSED'}
    print(f'closed: total {len(gts_closed)} samples')

    gts = {item['id']: [normalize_word(item['conversations'][1]['value'])] for item in gt}
    preds = {item['question_id']: [normalize_word(item['text'])] for item in pred}
    print(f'open and closed: {len(gts)} samples')
    eval_res = compute_scores(gts, preds)
    # gts_open = {item['id']: [normalize_word(item['conversations'][1]['value'])] for item in gt if item['answer_type'] == 'OPEN'}
    # preds_open = {item['question_id']: [normalize_word(item['text'])] for item in pred if item['answer_type'] == 'OPEN'}
    # print(f'open: {len(gts_open)} samples')
    # eval_res = compute_scores(gts_open, preds_open)
    for k, v in eval_res.items():
        print(f"{k}: {v:.3f}")

    # eval_res = compute_scores(gts_closed, preds_closed)
    # for k, v in eval_res.items():
    #     print(f"{k}: {v:.3f}")

    acc_res = compute_acc(choices_closed, gts_closed, preds_closed)
    print(f'acc = {acc_res}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # eval_model
    parser.add_argument("--model-path", type=str, default="/home/liusn/02resources/01output/qllava_wsi_output/qllava-lora-0516")
    parser.add_argument("--model-base", type=str, default="/home/liusn/02resources/02ckpt/vicuna-7b-v1.5") #
    parser.add_argument("--image-folder", type=str, default="/home/liusn/02resources/00dataset/TCGA-BRCA-patient-feats")
    parser.add_argument("--question-file", type=str, default="/home/liusn/02resources/00dataset/WsiVQA/WsiVQA_quilt_test_w_ans.jsonl")
    parser.add_argument("--answers-file", type=str, default="/home/liusn/02resources/01output/qllava_wsi_output/qllava-lora-0516/WsiVQA_quilt_test_pred.jsonl")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1") # vicuna_v1
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--answer-prompter", type=bool, default=False) # action="store_true"
    parser.add_argument("--single-pred-prompt", type=bool, default=True)

    # evaluate_metric
    # parser.add_argument("--gt", type=str, default="/home/liusn/02resources/00dataset/WsiVQA/WsiVQA_quilt_test_w_ans.jsonl")
    # parser.add_argument("--pred", type=str, default="/home/liusn/02resources/01output/qllava_wsi_output/qllava-lora-0516/WsiVQA_quilt_test_pred.jsonl")
    args = parser.parse_args()

    eval_model(args)

    # evaluate_metric(args)