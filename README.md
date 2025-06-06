# <div style="font-family:'Arial Black', sans-serif;font-size: 3em;font-weight: bold;background: linear-gradient(45deg, #FFD700, #FF8C00);-webkit-background-clip: text;background-clip: text;color: transparent;text-align: center;padding: 20px;text-shadow: 2px 2px 4px rgba(0,0,0,0.2);border-radius: 10px;"><span style="color: #333;">core</span><span style="background: linear-gradient(45deg, #FF8C00, #FF4500);-webkit-background-clip: text;background-clip: text;color: transparent;">Bagel</span></div>

**core Bagel** is a clone of the orignal Bagel Project.

This project does not aim at more functionality. *It hardens the core.*

<p align="center"><img src="assets/teaser.webp" width="95%"></p>


**core hardened features:**
- Cross-OS optimized
  - Works on Windows and Linux.
  - Easy unified install for all cards and OSes
- Performance optimization: 
  - Fully accelerated: all accelerators (triton, Flash-Attention) pre-built-in and fully enabled with custom optimized libraries.
  - Full support for all CUDA cards (yes, RTX 50 series Blackwell too)
  - Automatic VRAM optimization: Just start the app and it auto-configures the optimal settings based on your hardware.
    - Can run on 8GB VRAM (see Benchmark!). 
    - Configurable model-offload-placement: you can define where the disk offload-folder is placed.
  - Benchmarked for performance:
    - runs efficiently on 16GB VRAM already
- Flexibility:
  - Configurable Model placement: can  the models from anywhere you stored them.
  - Image export in png, jpg, webp
  - Fully offline: online links and loading of external web images removed from the app.



# Installation 

 
The installation in general consists of:

- Pre-Requisites: Check that your system can actually run the model
- Project Installation. It consists of 
    - cloning the repository
    - creating and activating a virtual environment
    - installing the requirements
    - getting the models (optionally re-using existing models)
    - starting the app.


## TLDR Installation

you need:
- python 3.12
- git
- CUDA drivers
- maybe CUDA toolkit


**Windows**
```
git clone https://github.com/loscrossos/core_bagel
cd core_bagel

py -3.12 -m venv .env_win
.env_win\Scripts\activate

pip install -r requirements.txt
```

**Linux**
```
git clone https://github.com/loscrossos/core_bagel
cd core_bagel

python3.12 -m venv .env_lin
. ./.env_lin/bin/activate

pip install -r requirements.txt
```

**All OSes**
You can use one of these steps (detailed steps below):
- **Option 1**: manualManual triggered Model Donwload: enter the `models` dir and use the `maclin_get_models.sh` or `win_get_models.bat`
- **Option 2**: reuse your models without changing their paths: run  `python appbagel.py --checkmodels` after install to generate `configmodels.txt` and edit the paths within the file. run the command again to verify it worked.

**Run the app**


Whenever you want to start the apps open a console in the repository directory, activate your virtual environment:

```
Windows:
.env_win\Scripts\activate
Linux:
. ./.env_lin/bin/activate
```


start the app with:

`python appbagel.py`


Stop the app pressing `ctrl + c` on the console



## Step-by-Step video guide

you can watch step-by-step guides for your OS. This is the same information as the next chapter.


OS	    | Step-by-step install tutorial
---	    | ---
Mac	    | todo
Windows	| todo
Linux  	| todo



## Step-by-Step install guide

### Pre-Requisites

In general you should have your PC setup for AI development when trying out AI models, LLMs and the likes. If you have some experience in this area, you likely already fulfill most if not all of these items. 


### Hardware requirements



**Installation requirements**

This seem the minimum hardware requirements:


Hardware    | **Mac** | **Win/Lin**
---         | ---     | ---
CPU         | n.a.    | 4Cores at least. CPU Will be used in full for final image processing. Still its not the main requirement. Any modern CPU should do 
VRAM        | n.a.    | Succesfully tested with 8GB VRAM (but its a pain). Tested comfortably with 16GB. The model needs 60GB VRAM to run at full speed.
RAM         | n.a.    | Uses some 2GB RAM (peak) during generation
Disk Space  | n.a.    | 27.5GB for the models






### Software requirements

**Requirements**

You should have the following setup to run this project:

- Python 3.12
- latest GPU drivers
- latest cuda-toolkit 12.8+ (for nvidia 50 series support)

I am not using Conda but the original Free Open Source Python. This guide assumes you use that.

**Automated Software development setup**

If:
- your pc is not setup for AI yet (python, latest CUDA installed and setup, code editors, sublibraries like ffmpeg, espeak)
- you want a fully free and open source, no strings attached, automated, beginner friendly but efficient way to setup a software development environment for AI and Python

you can use my other project: CrossOS_Setup, which setups your Mac, Windows or Linux PC automatically to be fully setup for AI Software Development. It includes a system checker to assess how well installed your current setup is, before you install anything:

https://github.com/loscrossos/crossos_setup

Thats what i use for all my development across all my systems. All my projects run out of the box if you PC is setup with it.
If you are already setup and happy thats ok. Its not a requirement. :)

### Project Installation

If you setup your development environment using my `Crossos_Setup` project, you can do this from a normal non-admin account (which you should actually be doing anyway for your own security).

Hint: "CrossOS" means the commands are valid on MacWinLin

 ---

Lets install core_bagel in 5 Lines on all OSes, shall we? Just open a terminal and enter the commands.



1. Clone the repo (CrossOS): 
```
git clone https://github.com/loscrossos/core_bagel
cd core_bagel
```

2. Create and activate a python virtual environment  

task       |  Windows                   | Linux
---        |  ---                       | ---
create venv|`py -3.12 -m venv .env_win`|`python3.12 -m venv .env_lin`
activate it|`.env_win\Scripts\activate`|`. ./.env_lin/bin/activate`

At this point you should see at the left of your prompt the name of your environment (e.g. `(.env_win)`)


3. Install the libraries (CrossOS):
```
pip install -r requirements.txt
```

Thats it.

---

Now we need the models...

### Model Installation

The needed models are about 27.6GB in total. You can get them in 2 ways:
- **use the model downloader**: Manual triggered Model Donwload: enter the `models` dir and use the `maclin_get_models.sh` or `win_get_models.bat`.
- **Re-use existing models**: re use models that you already downloaded


to see the status of the model recognition start any app with the parameter `--checkmodels`

e.g. `python appbagel.py --checkmodels`

The app will report the models it sees and quit without downloading or loading anything.



**Re-use existing models**


You can re-use your existing models by configuring the path in the configuration file `modelconfig.txt`.
This file is created when you first start any app. Just call e.g. `python appsbagel.py --checkmodels` to create it.
Now open it with any text editor and put in the path of the directory that points to your models. 
You can use absolute or relative paths. If you have a multiboot-Setup (e.g. dualboot Windows/Linux) you should use relative paths with forward slashes e.g. `../mydir/example`




**Checking that the models are correctly configured**

You can easily check that the app sees the models by starting any of the demos with the parameter `--checkmodels` and checking the last line ofthe output.
Even if some paths are missing thats ok as long as the last line says its ok.

e.g. `python appbagel.py --checkmodels`

```
[!FOUND!]: /Users/Shared/github/core_projectexample/models/somemodel/
[!FOUND!]: /Users/Shared/github/core_projectexample/models/someothermodel/
[!FOUND!]: /Users/Shared/github/core_projectexample/models/modeltoo/
----------------------------
FINAL RESULT: It seems all model directories were found. Nothing will be downloaded!
```


### Update

If you ever need to update the app
- because you know that the repository changed
- a bug got fixed

**update repository**

you can safely do so by starting a terminal in the repository directory and typing:
```
git pull
```
if you didnt change any original files this will safely update your app. Its ok to change configuration files (e.g. `configmodels.txt`) that were generated after cloning.

**update virtual environment**


If the requirements file changed you can safely update by deleting the old directory (`.env_mac/.env_win/env_lin`).

and recreating it using the steps above:
- create env
- activate env
- pip install

Normally you dont need this. you would read it explicitely. So like in 99% of cases you will not be doing this. 





# Usage 

You can use app as you always have. Just start the app and be creative!

## Starting the Apps


The app has the following name:

- `appbagel.py`

To start just open a terminal, change to the repository directory, enable the virtual environment and start the app. The `--inbrowser` option will automatically open a browser with the UI.

task         |  Windows                   | Linux
---          |  ---                       | ---
activate venv| `.env_win\Scripts\activate`|`. ./.env_lin/bin/activate`


for example (CrossOS)
```
python appbagel.py
```

A browser should pop up with the UI


To stop the app press `ctrl-c` on the console (CrossOS)






# Benchmark


### Setup

Benchmarks were run on:


I benchmarked my installation on:

&nbsp;  | MacOS     |Windows/Linux
---     | ---       | ---
CPU     | M1        |12Core
RAM     |16GB       |64GB       
GPU     |integrated | RTX 3060ti 8GB /RTX 5060ti 
VRAM    | unified   |8/16/24GB      
Storage | SSD       |NVME


### Results

Results mesures in s/it (seconds per iteration): less is better.

Lib   |MacOS  |Windows| Linux| VRAM Usage | RAM | CPU | swap
 ---  | ---   | ---   | ---  | ---        | --- | --- | ---
8GB   | x:xx  | 75sit | OOM  |  7.3GB     | 44GB| 100%| 20GB
16GB  | x:xx  | 4sit  | 7sit |  12GB      | 1GB | 20% | 2 GB


On 16GB VRAM it ran fine at 4sit. at 50 iterations per generation this means some 5 minutes per generation. The VRAM was 16GB but it only used only 12GB. So i think if you have a 12GB card you could be fine. However the memory optimization gets a threshhold at 12GB *free* VRAM.. so if you have less *free* VRAM you will fall into potato mode even if you have a 12GB card.


On 8GB VRAM it aborted on Linux with an Out of Memory (OOM) error. On Windows it went through but ran at 75sit, which means 1 hour per generation. At the same time it seems the accelerator decided to offload everything to RAM and disk and RAM usage went up to 44GB. Disk swap went to 20GB. So that 8GB card might need an update for this kind of models.

I am not sure why on the same hardware Linux OOMed. I suspect flash attention to need more tweaking on Linux. I compiled it myseld and might be looking into it maybe.


### Take away:


**If you have 16GB or more:**

You should be fine!

**If you have 12GB or more:**
Didnt test but i think you could be fine!


**if you have 8GB**:

It runs on windows.. more like crawls but yea... takes a long while.
I am not sure yet why it OOMs on Linux. it maybe the way i compiled flash-attention. I will be publishing a guide to compile and optimize Flash-attention so if you are on linux with 8GB stay tuned.

I will not be testing further but if you find a way let me know.


# Known Issues

Documentation of Issues i encountered and know of: None yet

 
### How to improve performance

based on my tests.

- if you have a 16GB card you should be ok.

- I didnt test 12GB cards but this could help:
  - 64GB of RAM will help if you have 
  - you can set the offload folder to your fastest drive.








## **Troubleshooting**

- If you face CUDA-related issues, ensure your GPU drivers are up to date.
- For missing models, double-check that all models are placed in the correct directories.


- if you have problems getting the softwar to run and you open an issue it is mandatory to include the output of 
```
python appbagel.py --sysreport
```




# Credits

**Original project page**

https://github.com/ByteDance-Seed/Bagel


**Original Authors**
> [Chaorui Deng](https://scholar.google.com/citations?hl=en&user=k0TWfBoAAAAJ), [Deyao Zhu](https://tsutikgiau.github.io/), [Kunchang Li](https://andy1621.github.io/), [Chenhui Gou](https://www.linkedin.com/in/chenhui-gou-9201081a1/?originalSubdomain=au), [Feng Li](https://fengli-ust.github.io/), [Zeyu Wang](https://zw615.github.io/), Shu Zhong, [Weihao Yu](https://whyu.me/), [Xiaonan Nie](https://codecaution.github.io/), [Ziang Song](https://www.linkedin.com/in/ziang-song-43b0ab8a/), Guang Shi :email: , [Haoqi Fan ](https://haoqifan.github.io/)

contact: shiguang.sg@bytedance.com



## License
core BAGEL is licensed under the Apache 2.0 licence, just as the original project


