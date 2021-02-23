#%% Set cwd
import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
os.getcwd()
# %% import libraries and modules
import importlib
import keras
import shutil
# import online learner modules:
import src.online_learning
importlib.reload(src.online_learning)
import src.utils
importlib.reload(src.utils)
from src.utils import dir_splitter
from src.online_learning import online_learner

# %% Define paths:
# all train data:
TRAIN_PATH_R = r"data_bsd\data_train"
# first randomized half of 360 test images
TEST_PATH = r"data_bsd/test"
# second randomized half of 360 test images
HOLDOUT_PATH = r"data_bsd/holdout"
# Path for all images trained on 
CURRENT_TRAIN_PATH = r"data_bsd/current_train_dir"

# %% # Create new folder with subsample of TRAIN_NR training images
#TRAIN_PATH_R_N = os.path.join(TRAIN_PATH_R, "N")
#TRAIN_PATH_R_P = os.path.join(TRAIN_PATH_R, "P")
all_img_list_N = os.listdir(r"data_bsd\data_train\N")
all_img_list_P = os.listdir(r"data_bsd\data_train\P")
nr_N = len(all_img_list_N)
nr_P = len(all_img_list_P)
nr = nr_N + nr_P

# Delete old CURRENT_TRAIN_PATH
shutil.rmtree(r"data_bsd/current_train_dir")
# Split train data to wished directory

# SET ON HOW MANY TRAIN IMAGES MODEL SHOULD BE TRAINED:
TRAIN_NR = 80

# Compute corresponding train share
train_share = (TRAIN_NR/0.8)/nr
# Split randomized TRAIN_NR images from all training data (with move)
dir_splitter(TRAIN_PATH_R, dest_path_1=CURRENT_TRAIN_PATH,
             dest_1_share=train_share)

#%% remove old CURRENT_TRAIN_DIR
#shutil.rmtree(CURRENT_TRAIN_DIR)
# %% Split all test data into two halfs (holdout and test):
try: 
    shutil.rmtree(r"data_bsd/test")
except:
    print("dir not existent!")
try: 
    shutil.rmtree(r"data_bsd/holdout")
except:
    print("dir not existent!")
    
dir_splitter(path=r'data_bsd\data_new',
             dest_path_1=HOLDOUT_PATH,
             dest_path_2=TEST_PATH,
             dest_1_share=0.5,
             dest_2_share=0.5)

#%% LOAD MODEL:
MODELDIR = r'keras_pretrained_model/best_CNN_2-dense-32-nodes-0.6-dropout'
CHDIR = os.path.join(dname, MODELDIR)
#print(CHDIR)
os.chdir(CHDIR)
#Load trained base:
MODELPATH = r'best_CNN_2-dense-32-nodes-0.6-dropout'
trained_base = keras.models.load_model(MODELPATH)
# Change directory back to repo
os.chdir(dname)
os.getcwd()

# %% Conduct online learning:
# delete ind_img, test and holdout if existent:
try: 
    shutil.rmtree(r"data_bsd/ind_img")
except:
    print("dir not existent!")

# retrain trained base:
online_learner(CURRENT_TRAIN_PATH, TEST_PATH, HOLDOUT_PATH,
               trained_embedder=trained_base,
               batch_size=16, dropout=0.6)
# Note: After each usage of online_lerner trained base has to be reloaded!
