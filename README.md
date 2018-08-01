# BIOMAG - 2018, Challenge
## Identifying seizures in MEG data: Localization of MEG seizures onset

This repository contains all the code necessary to reproduce our submission to
the [BIOMAG challenge of 2018](https://sites.google.com/view/biomag-2018-challenge/home).
The final slides can be found in `submission.pdf`

### Package Requirements

- [Picard (*44a13f0*)](https://pierreablin.github.io/picard/)
- [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/)
- [Nilearn](https://nilearn.github.io/)
- [**mne-python** (*v.0.16*)](https://mne-tools.github.io/0.16/index.html)
- [scikit-learn (patched version of *v.0.19.x*)](http://scikit-learn.org/0.19/documentation.html) 
check
[this](https://www.devroom.io/2009/10/26/how-to-create-and-apply-a-patch-with-git/)
in how to apply `0001-don-t-center-with-PCA.patch`

# How to run the code
Before running the code some data prepossessing is needed:
1. Download the challenge [data](https://box.bic.mni.mcgill.ca/s/Biomag2018),
    and set `mne_data_path` variable in `config.py` accordingly.

   *Alternatively you can use our copy of the data using [this dropbox link](https://www.dropbox.com/sh/8c46h5p98j3l0a6/AAB6_mnkCGY_tUP4RZswZ2IZa?dl=0)*
    
2. Create various surface reconstructions with FreeSurfer. The recommended FreeSurfer workflow is summarized on [this](https://surfer.nmr.mgh.harvard.edu/fswiki/RecommendedReconstruction) FreeSurfer's wiki page. Make sure to setup `subjects_dir` in `config.py` accordingly.

    *Alternatively you can use our precomputed surface reconstructions following [this dropbox link](https://www.dropbox.com/sh/po2u0ehdwm2wpue/AACJFvnZUWoogcXzQrAZkFdQa?dl=0)*
    
    
Once all data is in place, to reproduce the results just do:


```sh
$ ipython --matplotlib --gui=qt5 

```

```py
>>> %run run_mne_pipeline.py
```
