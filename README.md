# [ECCV24] LayeredFlow Benchmark

We introduce **LayeredFlow**, a real world benchmark with multi-layer annotations for optical flow of non-Lambertian objects, featuring 150k optical flow and stereo pairs across 185 diverse scenes with 360 unique objects.

LayeredFlow enables a new task called multi-layer optical flow. We also offer a large-scale, densely-annotated synthetic dataset with 60k images from 30 scenes for training. Fine-tuning on this dataset enhances the performance of existing optical flow methods on non-Lambertian objects.

If you find LayeredFlow useful for your work, please consider citing our academic paper:

<h3 align="center">
    <a href="https://arxiv.org/abs/2409.05688">
        LayeredFlow: A Real-World Benchmark for Non-Lambertian Multi-Layer Optical Flow
    </a>
</h3>
<p align="center">
    <a href="https://hermera.github.io/">Hongyu Wen</a>, 
    <a href="">Erich Liang</a>, 
    <a href="http://www.cs.princeton.edu/~jiadeng">Jia Deng</a><br>
</p>

```
@article{wen2024layeredflow,
    title={LayeredFlow: A Real-World Benchmark for Non-Lambertian Multi-Layer Optical Flow}, 
    author={Hongyu Wen and Erich Liang and Jia Deng},
    journal={arXiv preprint arXiv:2409.05688},
    year={2024},    
}
```

## Installation

```
conda env create -f env.yaml
conda activate layeredflow
```


## LayeredFlow Benchmark
<img src="imgs/benchmark_gallery.jpg" width='1000'>

### Download
Download the validation set (images + ground-truth) and test set (images) [here](https://drive.google.com/file/d/1EEFp7AE8ZX75ADztP74Mx7VZ6MOymneN/view?usp=sharing).

Unzip the data. Make soft links to the data under `RAFT/datasets` and `MultiRAFT/datasets` folder.
```
├── datasets
    ├── layeredflow
        ├── test
        ├── val
```

### Evaluation on Validation Set
You can find the pre-trained checkpoints [here](https://drive.google.com/drive/folders/1vB8wfeS1JpfTz-jKdiD_e_IM5Sn5Wztl?usp=sharing).

To evaluate RAFT on single layer subset of the benchmark using pre-trained checkpoints, download corresponding checkpoints and put them in `RAFT/checkpoints`, run
```
cd RAFT
python3 evaluate_firstlayer.py --checkpoint first_layer_S+L.pth --dataset layeredflow # first layer
python3 evaluate_lastlayer.py --checkpoint last_layer_L.pth --dataset layeredflow # last layer
```

To evaluate MultiRAFT on full set of the benchmark using pre-trained checkpoints, download corresponding checkpoints and put them in `MultiRAFT/checkpoints`, run
```
cd MultiRAFT
python3 evaluate.py --checkpoint MultiRAFT_S+L.pth --dataset layeredflow --mixed_precision
```

### Evaluation on Test Set
To evaluate your model on the test set and compare your results with the baseline, you need to submit your flow predictions to the [evaluation server](https://layeredflow.cs.princeton.edu).

Navigate to the RAFT directory and execute the following command to create your submission. A folder named "layeredflow_submission" will be created.
```
cd RAFT
python3 evaluate_firstlayer.py --checkpoint your_checkpoint --dataset layeredflow --create_submission # first_layer benchmark
python3 evaluate_lastlayer.py --checkpoint your_checkpoint --dataset layeredflow --create_submission # last_layer benchmark

cd MultiRAFT
python3 evaluate.py --checkpoint your_checkpoint --dataset layeredflow --create_submission # multi_layer benchmark
```

Submit your predictions to the evaluation server using the command below. Make sure to replace placeholders with your actual email, submission path, and method name:
```
python3 upload_submission.py --email your_email --path path_to_your_submission --method_name your_method_name --benchmark first_layer
```

Upon submission, you will receive a unique submission ID, which serves as the identifier for your submission. Results are typically emailed within 1 hour. Please note that each email user may upload only three submissions every seven days.

To make your submission public, run the command below. Please replace the placeholders with your specific details, including your submission ID, email, and method name. You may specify the publication name, or use "Anonymous" if the publication is under submission. It's optional to provide URLs for the publication and code.
```
python3 modify_submission.py --id submission_id --email your_email --anonymous False --method_name your_method_name --publication "your publication name" --url_publication "https://your_publication" --url_code "https://your_code"
```


## Synthetic Data Generation
<img src="imgs/synthetic_gallery.jpg" width='1000'>

### Pre-rendered Dataset
You can find the pre-rendered images [here](https://drive.google.com/drive/folders/17WFb_k5PNXb6C0b4I2Nwt1uzkQs3ctuW?usp=sharing).

To unzip the dataset, download all the files then run:
```
cat *.tar.gz.* | tar xvfz -
```

### Set up Blender
To generate multiple layer ground truth from blender, you will need to build customized blender.

First, clone the [Blender](https://github.com/blender/blender) source code to your machine. Using version 3.6 (v3.6.14 is tested for MacOS).
```
git clone --depth 1 --branch v3.6.14 git@github.com:blender/blender.git
```

Next, apply the [multi-layer ground truth patch](https://drive.google.com/file/d/1qozUApe_QqsqSwk4i08fd97-5a0-Snwx/view?usp=sharing).
```
git apply multiple-layer-ground-truth.patch 
```

Then build blender from source following [instructions](https://developer.blender.org/docs/handbook/building_blender).

### Rendering Scripts

Download [blenders files](https://drive.google.com/file/d/1Z7mgXQfUvg7rPXYc1nmUfkCEws-AGzlM/view?usp=share_link) and [hdrs files](https://drive.google.com/file/d/1-Nbe6i3fUI2vMxAeNjXnIMv8BwYJToE-/view?usp=share_link), save and extract them into `syn_datagen` folder. Then run `syn_datagen/generate.sh`.

## Acknowledgements

This project relies on code from existing repositories: [RAFT](https://github.com/princeton-vl/RAFT) and [ptlflow](https://github.com/hmorimitsu/ptlflow). We thank the original authors for their excellent work.