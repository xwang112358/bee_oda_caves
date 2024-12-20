# Bee's visual Acuity Collection

Visual acuity, the ability to perceive detail, is ecologically important, as it dictates what aspects of a visual scene an animal can resolve. The study of visual acuity in bee species helps biologists understand their evolutionary history and gain insight into their foraging strategies and navigation abilities. Sponsored by the Caves Lab, this project aims to design a pipeline that uses high-resolution 2D photos taken by the NSF-funded Big Bee Project to estimate visual acuity across different bee species. In the pipeline, we develop algorithmic approaches to measure the diameter of the ommatidia $D$ and estimate the interommatidial angles $\phi$ on the eyeʼs surface. By achieving a significant level of automation and accuracy, our pipeline will facilitate more efficient data collection for biologists.

Our codes are updated based on the open-source code provided in https://github.com/jpcurrea/ODA, which is the offcial implementation of Currea, J.P., Sondhi, Y., Kawahara, A.Y. et al. Measuring compound eye optics with microscope and microCT images. Commun Biol 6, 246 (2023).

Please check our results in our [poster](https://docs.google.com/presentation/d/15mhmS1tNokRFTDW90dSTaatf0L2uq_Fw_s_0KacIg5M/edit?usp=sharing).

<div align="center">
  <img src="pics/visual_acuity.png" width="50%">
</div>


## Prerequisite  
__Ommatidia Detection Algorithm & Contour Analysis__

```
conda create --name bee_oda python=3.9
```
```
conda activate bee_oda
```
```
pip install -r requirements.txt
```
We use `pytesseract==0.3.9` to analyze the scale bar. Please refer to this [page](https://github.com/h/pytesseract) for installation details and change this [line](https://github.com/xwang112358/bee_oda_caves/blob/24cd37c887c68cfc14238506414d311b841d7805/test_bee/funcs.py#L13) based on your OS.

Also, remember to download related packages for running jupyter notebook with these conda environments.

__Segment Anything(SAM)__

```
conda create -n sam_inf python==3.8
```
```
conda activate sam_inf
```

We deploy the newly released Segment Anything Model to help us segment the shape of bee's eyes with simple prompts, such as bounding box and points. It require a GPU with at least 8GB memory when making inferences with the SAM model. The code requires `python>=3.8, pytorch>=1.7` and `torchvision>=0.8`. Here is an example of setting up the SAM environment:

```
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

```
pip install -r test_bee/SAM/requirement.txt
```


- Download the [sam_vit_h_4b8939.pth](https://github.com/facebookresearch/segment-anything#model-checkpoints) and save it in `test_bee/SAM` folder
- Run `./test_bee/SAM/segment_eyes.ipynb` to segment the eyes and save the eye masks for further processing.

## Data 

Here, we provide 145 bee specimens collected at UCSB Cheadle Center. Each image was obtained by synthesizing multiple identical 6000*4000 pixel bee photos with varying focal points, allowing for the creation of high-resolution representations of different sections of the bee.

Please download the [data](https://drive.google.com/drive/folders/1Z8RyyXIZXyFs5L62kqnhB2llLOla0NY2?usp=sharing) and save it in the `test_bee/data` 
## Pipeline 

![Alt text](pics/pipeline.png)

```
cd test_bee
```
1. Detect ommatidia and sample diameters 
   ```
   python ommatidia_diam.py
   ```
   - input the catalog number of the bee, hef or hal
   - a window will pop up to select the sampling area on bee's eye
   - you can manually adjust the position of detected ommatidia on pop-up UI
   - output and save the distribution of ommatidia diameter in the selected area
2. Derive eye mask via SAM
   - run `./test_bee/SAM/segment_eyes.ipynb` to segment the eyes and save the eye masks for further processing.
3. Run `./test_bee/contour_analysis.ipynb` to analyze the eye masks and calculate the interommatidial angles.
4. The results are saved in `results/` folder.




## Methodologies


### Ommatidia Diameter Measurement 

![Alt text](pics/oda.png)

<!-- ![Alt text](pics/c2edaca972a2caedffecf1a0a24e756.png) -->

### Interommatidia Angles Measurement 

![interommatidial](pics/contour_analysis.png)
