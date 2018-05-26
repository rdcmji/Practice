import random

sample_num = 20000

trainmsg =[]
traindiff=[]
with open('train_msg.txt') as reader:
	trainmsg = reader.readlines()

with open('train_diff.txt') as reader:
	traindiff = reader.readlines()

rand_smpl = sorted(random.sample(range(len(trainmsg)), sample_num))
trainmsg = [trainmsg[i] for i in rand_smpl]
traindiff = [traindiff[i] for i in rand_smpl]

print (len(trainmsg), len(traindiff))
print (rand_smpl)

with open ('train.%d.msg'%(sample_num),'w') as writer:
	writer.writelines(trainmsg)


with open('train.%d.diff'%(sample_num), 'w') as writer:
	writer.writelines(traindiff)
