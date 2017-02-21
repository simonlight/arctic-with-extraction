# arctic-captions

Source code for [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](http://arxiv.org/abs/1502.03044)
runnable on GPU and CPU.

Joint collaboration between the Université de Montréal & University of Toronto.

## Dependencies

This code is written in python. To use it you will need:

* Python 2.7
* A relatively recent version of [NumPy](http://www.numpy.org/)
* [scikit learn](http://scikit-learn.org/stable/index.html)
* [skimage](http://scikit-image.org/docs/dev/api/skimage.html)
* [argparse](https://www.google.ca/search?q=argparse&oq=argparse&aqs=chrome..69i57.1260j0j1&sourceid=chrome&es_sm=122&ie=UTF-8#q=argparse+pip)

In addition, this code is built using the powerful
[Theano](http://www.deeplearning.net/software/theano/) library. If you
encounter problems specific to Theano, please use a commit from around
February 2015 and notify the authors.

To use the evaluation script (metrics.py): see
[coco-caption](https://github.com/tylin/coco-caption) for the requirements.

## Reference

If you use this code as part of any published research, please acknowledge the
following paper (it encourages researchers who publish their code!):

**"Show, Attend and Tell: Neural Image Caption Generation with Visual Attention."**  
Kelvin Xu, Jimmy Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Ruslan
Salakhutdinov, Richard Zemel, Yoshua Bengio. *To appear ICML (2015)*

    @article{Xu2015show,
        title={Show, Attend and Tell: Neural Image Caption Generation with Visual Attention},
        author={Xu, Kelvin and Ba, Jimmy and Kiros, Ryan and Cho, Kyunghyun and Courville, Aaron and Salakhutdinov, Ruslan and Zemel, Richard and Bengio, Yoshua},
        journal={arXiv preprint arXiv:1502.03044},
        year={2015}
    } 

## License

The code is released under a [revised (3-clause) BSD License](http://directory.fsf.org/wiki/License:BSD_3Clause).

## Implementation Update on 5/13/2016 by Lorne0

### Paper: http://arxiv.org/pdf/1502.03044v3.pdf
### Source code: 
* https://github.com/kelvinxu/arctic-captions
* Show, Attend and Tell の再現をやる http://penzant.hatenadiary.com/entry/2016/01/24/000000
* https://github.com/intuinno/arctic-captions
* https://github.com/Lorne0/arctic-captions

### Environment:
Please do it by yourself

### (I’ll take MSCOCO for example and run the codes in Lorne0/arctic-captions)
### Data preprocessing:
#### Download data
MSCOCO 
* http://mscoco.org/dataset/#download

Flickr30k

Flickr8k
* http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/Flickr8k_Dataset.zip
* http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/Flickr8k_text.zip

#### Download caffe model (just google these name, it’s easy to find):
* VGG_ILSVRC_19_layers.caffemodel
* VGG_ILSVRC_19_layers_deploy.prototxt
* (if need) VGG_ILSVRC_16_layers.caffemodel
* (if need) VGG_ILSVRC_16_layers_deploy.prototxt

#### Copy data in another dir, and copy the “preprocess.sh” to the new data dir, and run it to get 224x224 pictures

#### run ‘make_annotations.py’ to combine both captions in ‘captions.token’
* ex: COCO_train2014_000000517830.jpg#0	A stop sign and a lamp post on a street corner

#### After get ‘captions.token’, run ‘make_dic.py’ to get ‘dictionary.pkl’

#### Run ‘save_dic.py’ to get ‘capdict.pkl’(key:an image name, value: a list of captions)

#### Run ‘prepare_model_coco.py’ to get(need 7~8hrs to extract features):
* coco_align.test.exp1.pkl
* coco_align.train.exp1.pkl
* coco_align.val.exp1.pkl
* coco_feature.test.exp1.pkl
* coco_feature.train.exp1.npz
* coco_feature.val.exp1.pkl

###Training
Remember to change the model path in evaluate_coco.py, and run:
##### THEANO_FLAGS='mode=FAST_RUN,floatX=float32,device=gpu1' python evaluate_coco.py

###Evaluating
After get a ‘coco_deterministic_model.exp1.npz.pkl’, run:

##### python generate_caps.py -p 25 /path/coco_deterministic_model.exp1.npz ./result/res

where -p 25 means use 25 cores in parallel, be careful not to use too many cores

Get ‘res.dev.txt’ and ‘res.test.txt’

Get https://github.com/tylin/coco-caption first

python score.py dev /path/res.dev.txt >> score_result

python score.py test /path/res.test.txt >> score_result

#### Result

##### 10 epoch

./result/res.dev.txt
{'reflen': 52239, 'guess': [54180, 49180, 44180, 39180], 'testlen': 54180,
'correct': [34666, 15767, 6568, 2784]}
ratio: 1.03715614771
Bleu_1:	0.639830195644
Bleu_2:	0.452910759032
Bleu_3:	0.312423890302
Bleu_4:	0.215754258651
METEOR:	0.238656040704
ROUGE_L: 0.463549364456
CIDEr:	0.598491939189

./result/res.test.txt
{'reflen': 52201, 'guess': [54095, 49095, 44095, 39095], 'testlen': 54095,
'correct': [34614, 15665, 6608, 2887]}
ratio: 1.03628282983
Bleu_1:	0.639874295221
Bleu_2:	0.45184959727
Bleu_3:	0.312768371467
Bleu_4:	0.218021093333
METEOR:	0.238426572445
ROUGE_L: 0.464271760177
CIDEr:	0.614729348648



##### 17 epoch(early stop)
./result/res4.dev.txt
{'reflen': 55065, 'guess': [58795, 53795, 48795, 43795], 'testlen': 58795,
'correct': [36292, 16302, 6853, 2946]}
ratio: 1.06773812767
Bleu_1:	0.617263372736
Bleu_2:	0.432498636087
Bleu_3:	0.297274938528
Bleu_4:	0.205031588499
METEOR:	0.240483558079
ROUGE_L: 0.458247460903
CIDEr:	0.573334934357

./result/res4.test.txt
{'reflen': 54912, 'guess': [58841, 53841, 48841, 43841], 'testlen': 58841,
'correct': [36183, 16110, 6838, 3045]}
ratio: 1.07155084499
Bleu_1:	0.614928366275
Bleu_2:	0.428946842262
Bleu_3:	0.295336528977
Bleu_4:	0.205666987179
METEOR:	0.240799455851
ROUGE_L: 0.455100091911
CIDEr:	0.578968706505








