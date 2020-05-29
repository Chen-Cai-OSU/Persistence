
# Environment setup
I only have experience in MacOS and Ubuntu so it is recomended that you use these system if possible. For windows, I can also
try to help but my knowledge might be limited. 

Install [anaconda](https://docs.anaconda.com/anaconda/install/mac-os/). Take a look
at this [video](https://www.youtube.com/watch?v=uz6r0id2apA) if you run into some trouble.

To run the tutorial, you will need to install
* pip install matplotlib numpy
* Install [scikit-learn](https://scikit-learn.org/stable/install.html), [sklearn-tda](https://github.com/MathieuCarriere/sklearn-tda).
sklearn depends on gudhi so it is also suggested to install the gudhi (although you don't need to gudhi to run tutorial.py). You can install [gudhi](https://gudhi.inria.fr/#) from conda, see [here](https://anaconda.org/conda-forge/gudhi).
* (optional) it is recommended that you also install jupyter notebook. Take a look at 
this quick start [guide](https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/). 

IDE: You may consider install [Pycharm](https://www.jetbrains.com/pycharm/) as your IDE. Either professional or free community version is fine.

# Run the code

* It's good to know a bit command line basics. Check [this](https://lifehacker.com/a-command-line-primer-for-beginners-5633909) out.

* download the repository: ``git clone git@github.com:Chen-Cai-OSU/Persistence.git``

* go to the downloaded folder: ``cd Persistence``

* run the example: ``python tutorial.py``. I made a [video](https://www.dropbox.com/sh/bx7j4f2ql1unri6/AAAbegeJ6OHl8MDSL7EXuFEma?dl=0) explaining what the code is doing 
on the high level.

* You can also look at tutorial.ipynb. This is a jupyter notebook. If you can't open the file 
(github sometimes will be slow rendering the notebook, copy the link of notebook https://github.com/Chen-Cai-OSU/Persistence/blob/master/tutorial.ipynb at https://nbviewer.jupyter.org/)


# Tutorial for Clustering Persistence Diagrams
You can run either tutorial.py or tutorial.ipynb. The code are the same.

First go through tutorial.ipynb and make sure you understand the related concepts. 
Once you are confident about the all the functions (I added some comments for most functions), feel free to modify it such as changeing kernels 
or clustering algorithm.

# Contact
Contact me at cai.507@osu.edu if you have any questions. 