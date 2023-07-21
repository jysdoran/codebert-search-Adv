# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.
import logging
import sys,json
import numpy as np

def read_answers(filename):
    answers={}
    with open(filename) as f:
        for line in f:
            line=line.strip()
            js=json.loads(line)
            answers[js['url']]=js['idx']
    return answers

def read_predictions(filename):
    predictions={}
    with open(filename) as f:
        for line in f:
            line=line.strip()
            js=json.loads(line)
            predictions[js['url']]=js['answers']
    return predictions

def calculate_scores(answers,predictions):
    scores=[]
    for key in answers:
        if key not in predictions:
            logging.error("Missing prediction for url {}.".format(key))
            sys.exit()
        flag=False
        for rank,idx in enumerate(predictions[key]):
            if idx==answers[key]:
                scores.append(1/(rank+1))
                flag=True
                break
        if flag is False:
            scores.append(0)
    result={}
    result['MRR']=np.mean(scores)
    return result


def evaluate_predictions(answer_file, prediction_file):
    answers = read_answers(answer_file)
    predictions = read_predictions(prediction_file)
    scores = calculate_scores(answers, predictions)
    return scores


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate leaderboard predictions for NL-code-search-Adv dataset.')
    parser.add_argument('--answers', '-a', help="filename of the labels, in txt format.")
    parser.add_argument('--predictions', '-p', help="filename of the leaderboard predictions, in txt format.")

    args = parser.parse_args()
    scores = evaluate_predictions(args.answers, args.predictions)
    print(scores)

if __name__ == '__main__':
    main()
