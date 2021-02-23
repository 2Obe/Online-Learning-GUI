#%% import librarises
import shutil
import random
import os


# %% Directory splitter:
def dir_splitter(path, dest_path_1, copy=True, dest_path_2=None,
                 dest_1_share=0.5, dest_2_share=0.5):
    '''
    Splits all data of path-dir according to given shares to
    Parameter copy determines if images are moved (faster) or copied to dest_dir 
    destination dir 1 and 2 are destination directories to move copy to
    '''

    # Shuffle all images in P and N dir:
    p_files = os.listdir(path + "//P")
    n_files = os.listdir(path + "/N")
    random.shuffle(p_files)
    random.shuffle(n_files)

    print(p_files)
    print(n_files)

    # Create new N and P dir for dest_paths
    try:
        os.mkdir(dest_path_1)
        os.mkdir(dest_path_1 + "//P")
        os.mkdir(dest_path_1 + "//N")
    except:
        print("folders already exist!")
    if dest_path_2 is not None:
        try:
            os.mkdir(dest_path_2)
            os.mkdir(dest_path_2 + "//P")
            os.mkdir(dest_path_2 + "//N")
        except:
            print("folders already exist!")

    # Define number of p_files to split to dest_dir_1
    nr_p_1 = int(dest_1_share*len(p_files))
    # Define number of p_files to split to dest_dir_2
    nr_p_2 = int(dest_2_share*len(p_files))
    # Define number of n_files to split to dest_dir_1
    nr_n_1 = int(dest_1_share*len(n_files))
    # Define number of n_files to split to dest_dir_2
    nr_n_2 = int(dest_2_share*len(n_files))

    print("nr_p_1: ", nr_p_1)
    print("nr_n_1: ", nr_n_1)

    # Copy first dest_1_share * P_files to dest_1_P
    for p_f in p_files[:nr_p_1]:
        if copy:
            shutil.copy(path + "/P/" + p_f, dest_path_1 + "/P/" + p_f)
        else:
            shutil.move(path + "/P/" + p_f, dest_path_1 + "/P/" + p_f)
    # Copy first dest_1_share * N_files to dest_1_N
    for n_f in n_files[:nr_n_1]:
        if copy:
            shutil.copy(path + "/N/" + n_f, dest_path_1 + "/N/" + n_f)
        else:
            shutil.move(path + "/N/" + n_f, dest_path_1 + "/N/" + n_f)
    if dest_path_2 is not None:
        # Copy next dest_2_share * P_files to dest_2_P
        for p_f in p_files[nr_p_1:nr_p_1+nr_p_2+1]:
            if copy:
                shutil.copy(path + "/P/" + p_f, dest_path_2 + "/P/" + p_f)
            else:
                shutil.move(path + "/P/" + p_f, dest_path_2 + "/P/" + p_f)
        # Copy next dest_2_share * N_files to dest_2_N
        for n_f in n_files[nr_n_1:nr_n_1+nr_n_2+1]:
            if copy:
                shutil.copy(path + "/N/" + n_f, dest_path_2 + "/N/" + n_f)
            else:
                shutil.move(path + "/N/" + n_f, dest_path_2 + "/N/" + n_f)
