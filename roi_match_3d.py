from transformers import pipeline
from PIL import Image
import numpy as np
import cv2
import torch
from torch.nn.functional import cosine_similarity
from utils import Metric, Vis, Vis_cv2
import time
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random
device = 'cpu'

def