from os import listdir
import json

def rename_train(s):
    return 'COCO_train2014_'+'0'*(12-len(s))+s+'.jpg'

def rename_val(s):
    return 'COCO_val2014_'+'0'*(12-len(s))+s+'.jpg'

dic={}

with open('/tmp3/alvin/dataset/MSCOCO2014/captions_train2014.json') as fp:
    data = json.load(fp)

cnt=0
for i in range(len(data['annotations'])):
    s = rename_train(str(data['annotations'][i]['image_id']))
    if s not in dic:
        dic[s]=[]
        dic[s].append(data['annotations'][i]['caption'].replace('\n','').replace('"',''))
        cnt+=1
    else:
        #if len(dic[s])==5:
        #    continue
        dic[s].append(data['annotations'][i]['caption'].replace('\n','').replace('"',''))
        cnt+=1
print cnt

with open('/tmp3/alvin/dataset/MSCOCO2014/captions_val2014.json') as fp:
    data = json.load(fp)

for i in range(len(data['annotations'])):
    s = rename_val(str(data['annotations'][i]['image_id']))
    if s not in dic:
        dic[s]=[]
        dic[s].append(data['annotations'][i]['caption'].replace('\n','').replace('"',''))
        cnt+=1
    else:
        #if len(dic[s])==5:
        #    continue
        dic[s].append(data['annotations'][i]['caption'].replace('\n','').replace('"',''))
        cnt+=1
print cnt

cap = "/tmp3/alvin/arctic_data/coco/captions.token"
with open(cap,'w+') as fp:
    cnt=0
    for f in listdir('/tmp3/alvin/dataset/MSCOCO2014/train2014/'):
        for i in range(len(dic[f])):
            fp.write(f+'#'+str(i)+'\t'+dic[f][i]+'\n')
            cnt+=1
    for f in listdir('/tmp3/alvin/dataset/MSCOCO2014/val2014/'):
        for i in range(len(dic[f])):
            fp.write(f+'#'+str(i)+'\t'+dic[f][i]+'\n')
            cnt+=1
    print cnt


