We adapt the DialoGPT(https://github.com/microsoft/DialoGPT) model to be our chatbot model.
## Update
A new version is already implemented in branch "dev".
## Get started
#### Clone the repository
```
git clone https://github.com/jacksukk/Chatbot-Project.git
```
#### Corpus
https://github.com/facebookresearch/EmpatheticDialogues.git

#### Train
```
python train_c.py --emotion <emotion> --writer <tensorboard writer> --save <save path> --model <pretrained model> --ra <ratio between 2 loss> --inter <interlocutor you want to interact>
```

#### Test
```
python test_c.py --model <model> --filename <output file> --inter <interlocutor you want to interact>
```

#### Emotion Detector
please download the following link and put it in './' directory.
```
https://drive.google.com/file/d/1FZu2HIadORIvGD5nJAIOG6NjgtXrNXrz/view?usp=sharing
```
