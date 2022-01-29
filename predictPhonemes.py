import csv
import math
import os
import wave
from torch.utils import data
import librosa
import torch
import numpy as np
from progressbar import progressbar
from tqdm import tqdm
from torch.multiprocessing import Lock, Manager
from generators.librispeech import LRS2UnsupervisedLoader, LRS2AudioVisualPhonemeDataset
from models.audiovisual_model import FBAudioVisualCPCPhonemeClassifierLightning
from util.pad import audiovisual_batch_collate
from util.seq_alignment import beam_search

offsetMap={
	0:"I840",
	1:"I720",
	2:"I600",
	3:"I480",
	4:"I360",
	5:"I240",
	6:"I060",
	7:"base",
	8:"B060",
	9:"B240",
	10:"B360",
	11:"B480",
	12:"B600",
	13:"B720",
	14:"B840",
	15:"jumble"
}

def levenshtein(a, b):
    "Calculates the Levenshtein distance between a and b."
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a, b = b, a
        n, m = m, n

    current = list(range(n+1))
    for i in range(1, m+1):
        previous, current = current, [i]+[0]*n
        for j in range(1, n+1):
            add, delete = previous[j]+1, current[j-1]+1
            change = previous[j-1]
            if a[j-1] != b[i-1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]

def createDatasetPaths():
	paths=[]

	for x in range(6):
		for y in range(28):
			for key in offsetMap:
				paths.append("speaker"+str(x+1)+"clip"+str(y+1)+offsetMap[key])

	return paths

#Get a full list of all videos with speakers, sentences, and offsets
per_ckpt = "/home/analysis/Documents/studentHDD/chris/lightning_logs/version_0/checkpoints/epoch=10-step=63018.ckpt"
dest_csv="~/Documents/studentHDD/chris/predicitiveCodingExperiment/predicitiveCodingResults.csv"
datasetPath="/home/analysis/Documents/studentHDD/chris/monoSubclips/"

model = FBAudioVisualCPCPhonemeClassifierLightning(src_checkpoint_path=per_ckpt, batch_size=1, cached=False, LSTM=True).cuda()
per_checkpoint = torch.load(per_ckpt)

model.load_state_dict(per_checkpoint['state_dict'])

avgPER = 0
nItems = 0

downsamplingFactor = 160

test_params = {'batch_size': 1,
	'shuffle': False,
	'num_workers': 3}

model.eval()

testIDs=createDatasetPaths()

testSet=LRS2AudioVisualPhonemeDataset(testIDs, datasetPath, test_params['batch_size'])
testGenerator = data.DataLoader(testSet, collate_fn=audiovisual_batch_collate, **test_params)

for index, data in tqdm(enumerate(testGenerator), total=len(testGenerator)):
	i = index
	print(type(i))
	with torch.no_grad():
		x_audio, x_visual, y = data
		x_audio = x_audio.cuda()
		x_visual = x_visual.cuda()
		y = y.squeeze()
		y_hat = model.get_predictions((x_audio, x_visual)).squeeze()

		predSeq = np.array(beam_search(y_hat.cpu().numpy(), 10, model.phoneme_criterion.BLANK_LABEL)[0][1], dtype=np.int32)

		y = y.numpy()

		#per = levenshtein(y, predSeq)/y.shape[-1]

		#writer.writerow([index, per])

		#avgPER += per
		#nItems += 1