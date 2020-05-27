# Art Style Transfer
Using pretrained art style transfer image nets, make your regular photos artistic!

## Installation 
```bash
pip install imutils
pip install cv2
```

## Usage

File ```neural_style_transfer.py``` contails imports for the library and the function to convert the supplied image to art image. 

```python
img = style_trans("models/instance_norm/starry_night.t7", "images/Rich.jpg")
cv2.imshow("Output", img)
cv2.waitKey(0)
```

[Source](https://www.pyimagesearch.com/2018/08/27/neural-style-transfer-with-opencv/)
