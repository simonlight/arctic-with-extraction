import cPickle
import json
import sys
sys.path.append('/home/extra/b02902030/coco_eval/coco-caption/')
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor

def score(ref,hypo):
    scorers = [
        (Bleu(4),["Bleu_1","Bleu_2","Bleu_3","Bleu_4"]),
        (Meteor(),"METEOR"),
        (Rouge(),"ROUGE_L"),
        (Cider(),"CIDEr")
    ]
    final_scores = {}
    for scorer,method in scorers:
        score,scores = scorer.compute_score(ref,hypo)
        if type(score)==list:
            for m,s in zip(method,score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    print 'Bleu_1:\t',final_scores['Bleu_1']  
    print 'Bleu_2:\t',final_scores['Bleu_2']  
    print 'Bleu_3:\t',final_scores['Bleu_3']  
    print 'Bleu_4:\t',final_scores['Bleu_4']  
    print 'METEOR:\t',final_scores['METEOR']  
    print 'ROUGE_L:',final_scores['ROUGE_L']  
    print 'CIDEr:\t',final_scores['CIDEr']  

#sys.argv[1]='test'or'dev'

fpath = sys.argv[2]
print sys.argv[2]

with open('capdict.pkl') as fp:
    dic = cPickle.load(fp)

if sys.argv[1]=='test':
    fpp = './splits/coco_test.txt'
else:
    fpp = './splits/coco_val.txt'

tmp=[]
with open(fpp) as fp:
    for f in fp:
        tmp.append(dic[f.strip('\n')])
    ref={idx:rr for idx,rr in enumerate(tmp)}

tmp=[]
with open(fpath) as fp:
    for f in fp:
        tmp.append([f.strip('\n')])
    hypo={idx:rr for idx,rr in enumerate(tmp)}

'''
with open('cap.json') as fp:
    t = json.load(fp)
    tmp = []
    for i in range(6513,7010):
        s = 'video'+str(i)
        tmp.append(t[s])
    ref = {idx:rr for idx,rr in enumerate(tmp)}

with open(fpath) as fp:
    d = json.load(fp)
    del d['video6510']
    del d['video6511']
    del d['video6512']
    for i in range(6513,7010):
        s='video'+str(i)
        d[s] = [' '.join(d[s])]
    tmp = []
    for i in range(6513,7010):
        s = 'video'+str(i)
        tmp.append(d[s])
    hypo = {idx:rr for idx,rr in enumerate(tmp)}
'''

#print ref.keys()
#print hypo.keys()
#print len(ref.keys())
#print len(hypo.keys())
score(ref,hypo)
print ''
