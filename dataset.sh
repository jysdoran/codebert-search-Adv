#!/bin/bash
wget https://github.com/microsoft/CodeXGLUE/raw/main/Text-Code/NL-code-search-Adv/dataset.zip \
&& unzip dataset.zip \
&& cd dataset \
&& wget https://zenodo.org/record/7857872/files/python.zip \
&& unzip python.zip \
&& python preprocess.py \
&& rm -r python \
&& rm -r *.pkl \
&& rm -r *.txt \
&& rm python.zip \
&& cd ..
&& rm dataset.zip