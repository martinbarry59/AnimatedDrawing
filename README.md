# Animated drawing live (inspired from [Meta](https://github.com/facebookresearch/AnimatedDrawings/tree/main))
This repo is inspired by animted drawing by Meta. 
We used part of their code and the algorithm from [A Method for Animating Children's Drawings of the Human Figure](https://dl.acm.org/doi/10.1145/3592788).

## Installation
*This project has been tested with  Ubuntu 24.04. If you're installing on another operating system, you may encounter issues.*

We *strongly* recommend activating a Python virtual environment prior to installing Animated Drawings Live.
we strongly recommand the use of the great [uv](https://github.com/astral-sh/uv) python library manager based on Rust. Then run the following commands:

````bash
    git clone git@github.com:martinbarry59/AnimatedDrawingLive.git
    # create and activate setup the virtualenv
    uv sync
    uv build
    uv pip install -e . 
````

To be able to annotate the image you need to install your own container (we give you the .def file used for our own container). We locally used [Apptainer](https://apptainer.org/docs/admin/main/installation.html).
if using Apptainer please use
````bash
  sudo apptainer build my_animated.sif my_animated.def
````


### Quick Start
To get started, follow these steps:
1. Open a terminal and activate the animated_drawings conda environment:
````bash
~ % source .venv/bin/activate
````


2. We set up a pipeline directory that is end to end reading an image (put the image in the folder pipeline/images/) and animate it live to use it please run
   ````bash
   ~% cd pipeline
   ~% source pipeline.sh
