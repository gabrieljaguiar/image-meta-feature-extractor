# Image Meta-feature Extractor

Library to extract meta-features from images.

## Instalation

```bash
git clone https://github.com/gabrieljaguiar/image-meta-feature-extractor.git
```

## Instructions

The input folder is the folder of every image the user wants to extract the meta-features. The images can be in any format that opencv can open. The output file must be a .csv file in which the features are going to be write. There is a folder with example files, please check them.

```bash
python run.py ./example/input_folder/ ./example/output/output.csv/
```

## Meta-Features

In this library, 97 meta-features are extracted.

- Statistical (3)
- Colour-based (36)
- Histogram (21)
- Border (16)
- Image Quality (2)
- Texture (19)




## Requirements

The following Python packages are required:

- numpy
- pandas
- opencv2
- scikit-image
- imutils

Also, use Python **3.6**+!

## Citation
All of these features are presented or referenced in Aguiar _et al_ (2019) **[1]**. Also, if you use this extractor, please cite us:

```
@article{aguiar2019meta,
  title={A meta-learning approach for selecting image segmentation algorithm},
  author={Aguiar, Gabriel Jonas and Mantovani, Rafael Gomes and Mastelini, Saulo M and de Carvalho, Andre CPFL and Campos, Gabriel FC and Junior, Sylvio Barbon},
  journal={Pattern Recognition Letters},
  volume={128},
  pages={480--487},
  year={2019},
  publisher={Elsevier}
}
```
