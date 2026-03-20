# README for ABGABE

## task
You are asked to hand in a project, as we have set up in the class in the geo-oma-24 folder, and you can easily built on that and extend it. Since evaluating code in 2026 is not the smartest way to teach, i want you to demonstrate that you deeply understand the concepts we have talked about, still in a hands on approach.

For the hands on part, you can either work on your "own" DL project, or simply extend the one we have set up. This is on you, the requirements of the assignment apply to both cases. Either way, the hands on task should be designed as such:

    implement a the pytorch based DL-pipeline.
    choose a baseline to compare against, and implement it.
    modify the baseline in a way you think is interesting/helping to perform better on your model, and implement it. I would recommend to focus on one of the typical building blocks in our pipeline, but sure you can also tackle all, if you have the time.
    run experiments to compare the modified version with the baseline. And provide a brief report.

If you modify the SLURM based workflow of our class project, or if you do everything in jupyter notebooks and interactive sessions, is on you. I recommend to use the SLURM workflow, since it is more realistic and you will get more familiar with it, but if you want to do it in a more interactive way, that is also fine. The important thing is that you understand the concepts and can demonstrate that understanding in your implementation and report. I will not evaluate the coding part.

The final project folder should contain:

    Pipeline Overview In e.g. the README.md, describe the typical DL-pipeline we have talked about over and over during the class, now is the time you write it down yourself. Even better, draw the major components in a diagram and in text you can focus on their details. The pipeline description should be general, not yet related or focused on your task.

    Brief Problem Description Now introduce the problem at hand, very briefly. Input data and which goal you aim to achieve overall, not the details you focus on later.

e.g. "I have a set of Sentinel-1 images showing offshore infrastructure, and I want to detect the presence of offshore wind turbines in these images and classify their deployment status."

You can be more verbose if you like, also providing some images to describe the data and goal is always a good idea.

    Baseline Description Introduce a baseline setup, your standard implementation of a DL-pipeline. Follow our typical structure which we have discussed in the class over and over.

    Motivation for Modification Now briefly describe on which problem you want to focus on, e.g. in relation to observed baseline performance, or an idea you came up with from a conceptual standpoint. Answer the question: What limitation of the baseline are you addressing?

    Implementation Description Describe how you implemented the modification, e.g. in the last section you said something like, "Classes are unbalanced in the dataset, leading to poor performance on the minority class, therefore I want to implement a weighted loss function to address this issue." Now you can describe how you implemented the weighted loss function, and how it fits into the overall pipeline. e.g. "I go for a weighted cross entropy loss, where the weights are calculated based on the class distribution in the training set. I implemented this in the loss function module of my pipeline, and it is called during the training loop as usual."

You should be a bit more verbose here, but the important thing is that you clearly describe what you did and how it fits into the overall pipeline, its not necessary to show code here.

    Experiments Run at least two experiments comparing your adjustment with the baseline. You are invited to run more, e.g. in relation to the above example-motivation, show different combinations of weight values for the loss.

Provide at least loss curves for the trainings, for train and val. If you want to report other metrics, feel free to do so. You can create nice plots from the available data in the tf.event files, but it is also fine for me if you simply use screenshots of the tensorboard in this class.

    Results and Discussion Show the comparison of your baseline to your adjustment(s) and shortly discuss why the result looks like it does, if the outcome is worse, this is no problem :) I want you to show that you can run experiments not to process perfect results.


## ABGABE

### Pipeline overview

### Problem description

Cloud detection in multispectral satellite imagery is a base problem in the literature. 

### Implementation

#### Dataset
 1. SatelliteCloudGenerator on scribble
 2. validation via cloudsen12 high

#### HPO
Because the task is not a trivial one and first trainings on default-like parameters showed sub-par results, hyperparameter-optimization was considered. It was implemented using the optuna framework.
To be compatible with SLURM, a existing *Trainer class as extended with the `setup` method, which is called by optuna, to setup a study. Then, in the `__call__` method, a study is created and passed on to be executed as a SLURM job.

### Experiments
HPO was run with following parameters. The weights for classes were arbitrarily chosen, but on the grounds that `thin` and `shadow` appear less than `thick` who appear less than `clear`. This information was gained by analysing the unbalanced results of dice losses from previous, smaller experiments and based on the probabilities used in the synthetical data generation.
```bash 
python3 submit_hpo.py --user di54xat --n-trials 100 --epochs 5 --experiment-id hpo_006_cs12val_nt100-weights1244_epochs5 --aspp --class-weights 1.0 2.0 4.0 4.0 --use-cloudsen12-validation
```

### Training

### Results & Discussion
