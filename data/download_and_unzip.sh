#!/bin/bash
fileid=1Ks8wcBP10tyHwdzjIt_J8mnHiFAm3wwj
gdown https://drive.google.com/uc?id=${fileid}

unzip -qq tiny-imagenet-200-ttt.zip
rm -f tiny-imagenet-200-ttt.zip
