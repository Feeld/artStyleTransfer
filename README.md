# Art Style Transfer
Using pretrained art style transfer image nets, make your regular photos artistic!

## Installation 
```bash
pip install imutils
pip install opencv-python
pip install numpy
pip install PIL
```

## Usage

File ```neural_style_transfer.py``` contails imports for the library and the function to convert the supplied image to art image. 

```python

pil_image = Image.open("images/Feeld.jpg") # open image using PIL
img = style_trans("models/instance_norm/mosaic.t7", pil_image)
display(im)

```

[Source](https://www.pyimagesearch.com/2018/08/27/neural-style-transfer-with-opencv/)

[License](https://github.com/sdujump/fast-neural-style-1)

The pretrained models come from the research paper written by [Johnson et al. (2016)](https://cs.stanford.edu/people/jcjohns/eccv16/). For commercial use of them, we need to reach out to them. 
