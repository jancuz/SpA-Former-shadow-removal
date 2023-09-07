# Our project - Role of shadows in object recognition task

by Yana Kuznetcov

* [Main repository](https://github.com/jancuz/ShadowProject.git)
* Shadow amount calculation and visualization
* Correlation analysis btw. shadow amount and prediction confidence of the robust models

## based on SpA-Former: An Effective and Lightweight Transformer for Image Shadow Removal

by Xiao Feng Zhang, Chao Chen Gu, and Shan Ying Zhu [[paper link](https://arxiv.org/pdf/2206.10910.pdf)]
forked from [original repository](https://github.com/zhangbaijin/SpA-Former-shadow-removal.git)

### Citations
```
@article{zhang2022spa,
  title={SpA-Former: An Effective and Lightweight Transformer for Image Shadow Removal},
  author={Zhang, Xiao Feng and Zhao, Yu Di and Gu, Chao Chen and Lu, Chang Sheng and Zhu, Shan Ying},
  journal={arXiv e-prints},
  pages={arXiv--2206},
  year={2022}
```

# Train, Test, Evaluate, useful Data, ...

See [original repository](https://github.com/zhangbaijin/SpA-Former-shadow-removal.git)

## 1. SpAFormer to detect shadows

### Trained on grayscale (GS) images from ISTD, SOBA datasets

* [config.yml](https://github.com/jancuz/SpA-Former-shadow-removal/blob/main/results/000009-train%20with%20GS%20test%20on%20GS%20ISTD%20part/config.yml) used for training on GS ISTD
* [config.yml](https://github.com/jancuz/SpA-Former-shadow-removal/blob/main/results/000033-train%20on%20GS%20test%20on%20GS%20on%20SOBA%20bw%20masks/config.yml) used for training on GS SOBA
  
Modify the `config.yml` with the parameters used in `config.yml` files described higher and run:

```bash
python train.py
```

Here you can find [pre-trained model on GS ISTD](https://drive.google.com/file/d/1-MrP7dV4KTDt7IdJZBAkFeRv88snDV3u/view?usp=drive_link) and [pre-trained model on GS SOBA](https://drive.google.com/file/d/1-Rm6qEKhLUe30G8CMq8UH1t_e5NJkfak/view?usp=drive_link).

## 2. Test on ImageNet

First, the dataset is trained on 640x480, so you should resize the test dataset, you can use the code to resize your image 
```bash python bigresize.py```
and then follow the code to test the results:
```bash
python predict.py --config <path_to_config.yml_in_the_out_dir>
                  --test_dir <path_to_a_directory_stored_test_data>
                  --out_dir <path_to_an_output_directory>
                  --pretrained <path_to_a_pretrained_model> --cuda
```

Or you can use ```demo.py```
* to make the prediction and save it,
* to collect statistics for the model (BER calculation),
* to calculate the shadow amount

```bash
python demo.py --root_path <path_to_data>
               --shadow_count <True/False>
               --GT_access <True/False - BER_calculation_if_GT_data_is_available>
               --pretrained <path_to_a_pretrained_model>
               --save_filepath <path_to_an_output_directory>
               --save_shadow_info_rgb <path_to_a_txt_file_with_shadow_amount_info>
               --save_res_imgs<True/False save_output_or_not>
```

* There are [results](https://drive.google.com/file/d/1-q4miwXtkeKPlLcY4xt4ndqI5SpJEKH8/view?usp=drive_link) on ImageNet val. dataset with model trained on GS ISTD

<p align="center"><img src="imgs/ISTD GS ImageNet val.png">
  
* There are [results](https://drive.google.com/file/d/1-vxJZ3MygMpN4PCE_tgc1DDAlHJ5TF9O/view?usp=drive_link) on ImageNet val. dataset with model trained on GS SOBA

<p align="center"><img src="imgs/SOBA GS ImageNet val.png">


