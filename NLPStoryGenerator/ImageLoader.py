# import skimage.io as io
import random
import imagecodecs
import os
from pyparsing import Word
from torch import tensor
import torch
import numpy as np
import PIL
from PIL import Image, ExifTags
from PIL.ExifTags import TAGS
import chardet
import ProcessText
from collections import Counter
from itertools import chain

PIL.Image.MAX_IMAGE_PIXELS = np.inf
# dtype = torch.cuda.FloatTensor 
def load_image(filepath: str):
    # shape-(Height, Width, Color)
    # if(filepath[-3:] != "tif"):
    #     image = io.imread(str(filepath))
    # else:
    image = imagecodecs.imread(str(filepath))
    if image.shape[-1] == 4:
        # Image is RGBA, where A is alpha -> transparency
        # Must make image RGB.
        image = image[..., :-1]  # png -> RGB
    return image

def load_image_metadata(filepath: str):
    image = Image.open(filepath)
    
    exif = { ExifTags.TAGS[k]: v for k, v in image.getexif().items() if k in ExifTags.TAGS }
    desc = None
    if "ImageDescription" in exif:
        desc = exif["ImageDescription"]
        if(type(desc) is not str):
            the_encoding = chardet.detect(desc)
    return desc
    # print("done")

def load_path_metadata(filepath: str):
    return {path : ProcessText.clean(load_image_metadata(path)) for path in get_filepaths(filepath)}
    
def filenames_from_dict(names_and_numbers: dict, folder="images"):
    out = []
    for name in names_and_numbers.keys():
            for i in range(1, names_and_numbers[name] + 1):
                out.append(f"{folder}/{name}_{i}.jpg")
    return out
def get_filepaths(path: str):
    return [path + "/" + out for out in os.listdir(path)]
def save_npys(paths: "list[str]", filepath):
    for i in range(len(paths)):
        img = load_image(paths[i])
        if(len(img.shape) == 3 and img.mean() <= 253):
            np.save(f"{filepath}_{i}.npy", img)
        else:
            img = ((img.astype('float32')/65535) * 255).astype('uint8')
            np.save(f"{filepath}_{i}.npy", img)
def filter_bad_images(paths: "list[str]", filepath):
    for i in range(len(paths)):
        img = load_image(paths[i])
        if(len(img.shape) == 3 and img.mean() <= 253):
            np.save(f"{filepath}_{i}.npy", img)
        else:
            print(img.shape, img.mean, i)
def load_mmaps(filepath: str):
    maps = []
    i=0
    try:
        while True:
            maps.append(np.load(f"{filepath}_{i}.npy", mmap_mode='r'))
            i += 1
    except:
        return maps
    
def get_word_data(caption_list: "list[str]", k=np.inf):
    # all_words = "".join(chain.from_iterable(caption_list))
    # bag_of_words = ProcessText.create_bag_of_words(all_words, 10000)
    tokenized_captions = Counter(chain.from_iterable([set(ProcessText.tokenize(caption)) for caption in caption_list]))
    doc_count = len(caption_list)
    idfs = ProcessText.InverseFrequency(tokenized_captions, doc_count)
    return idfs
# print(load_image("TelescopeClassifier\HubbleImages\heic0108a.tif").shape)
class WordCaptionThing:
    def __init__(self, n=-1) -> None:
        if(n > 0):
            self.data =  load_path_metadata("HubbleImages")
            for _ in range(len(self.data) - n):
                self.data.pop(random.choice(list(self.data.keys())))
        else:
            self.data = load_path_metadata("HubbleImages")
        self.idfs = get_word_data(list(self.data.values()))
        self.special_words = ["red", "crimson", "orange", "yellow", "green", "blue", "white", "black", "light", "dark", "iridescent", "bright", "brightly", "glowing", "pale", "colossal", "large", "faint", "faintly", "small", "tiny", "dense", "denser", "cloud", "clouds", "flame"]        
        
        # print([word in self.idfs for word in self.special_words])
        for word in self.special_words: self.idfs[word] *= 5

    def get_random_prompt(self, k: int):
        prompt_path = random.choice(list(self.data.keys()))
        tokenized_caption = ProcessText.tokenize(self.data[prompt_path])
        weights = [self.idfs[word] if word in self.idfs else 0 for word in tokenized_caption]
        
        if(len(weights) > k):
            windows = np.sum([weights[i:-1] for i in range(k)], axis=0)
            ind = np.argmax(windows)
            if (ind < k):
                ind = k
            prompt = tokenized_caption[(ind - k):(ind + 1)]
            if(len(prompt) < 1):
                print("uh oh")
            out = " ".join(prompt)
            
            return prompt_path, out
        return None
        print("Done!")
    
    def __str__(self):
        return "\n\n".join(list(self.data.values()))