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

#Get a full list of all videos with speakers, sentences, and offsets
per_ckpt = "/home/analysis/Documents/studentHDD/chris/lightning_logs/version_0/checkpoints/epoch=10-step=63018.ckpt"
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

paths=[]

for x in range(6):
	for y in range(28):
		for z in range(16):
			paths.append("speaker"+str(x+1)+"clip"+str(y+1)+offsetMap[z])

# for path in paths:
# 	print(path)

model = FBAudioVisualCPCPhonemeClassifierLightning(src_checkpoint_path=per_ckpt, batch_size=1, cached=False, LSTM=True).cuda()
per_checkpoint = torch.load(per_ckpt)

model.load_state_dict(per_checkpoint['state_dict'])

for x in tqdm(range(len(paths))):
	visualInput=np.load("/home/analysis/Documents/studentHDD/chris/monoSubclips/"+paths[x]+".npy")
	audioInput=wave.open("/home/analysis/Documents/studentHDD/chris/monoSubclips/"+paths[x]+".wav","r")