# %% import necessary packages
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import shutil
import os
from PIL import ImageTk
from PIL import Image
import cv2
from tkinter import Label, Button
import tkinter as tk
from sklearn.metrics import classification_report
from src.embedder import EmbedderModeler



# %% Show image in GUI
def displayer(file_list):
    '''
    Creates GUI for Labeling images in a given filepath-list (file_list)
    '''
    # Create new empty panel (to display images)
    panelA = None
    # Create new empty widget
    #root = tk.Toplevel()
    root = tk.Tk()
    # Introduce variable which is only changed if loop is broken
    broken = 0
    # Set color of widget in KIT style
    root.config(bg="#F3F2F1")
    #root.withdraw()
    # Define variable to store if a button was clicked:
    clicked = tk.IntVar()
    stop = tk.IntVar()
    retrain = tk.IntVar()

    # Define filelist with files containing 0 labeled imgs:
    files_list_N = []
    files_list_P = []

    #print("clicked at beginning: ", clicked.get())

    img_path = ""

    def submit_0():
        # Append variable to files_list_0
        files_list_N.append(img_path)
        # Change variable:
        clicked.set(1)
        # print("submitted")

    def submit_1():
        # Append variable to files_list_0
        files_list_P.append(img_path)
        # Change variable:
        clicked.set(1)
        # print("submitted")

    # Define stop variable

    def destr():
        # set clicked variable to end current iteration
        clicked.set(1)
        # set click variable to break loop and avoid start of next iteration
        stop.set(1)

    def posttrain():
        # Set variable for retraining the model to 1
        retrain.set(1)
        # exit window and go on with retraining
        destr()

    # Load button images:
    b_img_1 = tk.PhotoImage(file=r"GUI/Label_1_n.png")
    b_img_2 = tk.PhotoImage(file=r"GUI//Label_0_n.png")
    b_img_3 = tk.PhotoImage(file=r"GUI/exit_n.png")
    b_img_4 = tk.PhotoImage(file=r"GUI/retrain_n.png")

    # Create buttons for labeling in the GUI
    btn_1 = Button(root, bd=0, bg="#F3F2F1", command=submit_1, image=b_img_1)
    btn_2 = Button(root, bd=0, text="0", bg="#F3F2F1",
                   command=submit_0, image=b_img_2)
    btn_3 = Button(root, bd=0, text="exit", bg="#F3F2F1",
                   command=destr, image=b_img_3)
    btn_4 = Button(root, bd=0, text="retrain", bg="#F3F2F1",
                   command=posttrain, image=b_img_4)
    # Pack buttons on widget (root)
    btn_3.pack(side="bottom", fill="y", expand="no", padx="10", pady="10")
    btn_4.pack(side="bottom", fill="y", expand="no", padx="10", pady="10")
    btn_2.pack(side="bottom", fill="y", expand="no", padx="10", pady="10")
    btn_1.pack(side="bottom", fill="y", expand="no", padx="10", pady="10")

    for img_path in file_list:
        image = cv2.imread(img_path)
        # print(image)
        # Convert img to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # convert the images to PIL format...
        image = Image.fromarray(gray)
        # ...and then to ImageTk format
        image = ImageTk.PhotoImage(image, master=root)
        #print("image after conversion: ", type(image))
        # Initialize panel in first iteration
        if panelA is None:
            # the first panel will store our original image
            panelA = Label(root, bg="#F3F2F1", image=image)
            panelA.image = image
            panelA.pack()

            # kick off the GUI
            root.update_idletasks()
            root.update()
            #print("clicked after first image: ", clicked.get())
            # Wait until button is clicked:
            root.wait_variable(clicked)
            # print("next!")
            # if exit button is clicked leave current widget and break loop
            if stop.get() == 1:
                broken = 1
                break

        # otherwise, update the image panels
        else:
            # update the pannels
            panelA.configure(image=image)
            panelA.image = image

            # kick off the GUI
            root.update_idletasks()
            root.update()

            # Wait until button is clicked:
            # print("waiting!")
            root.wait_variable(clicked)
            # print("next!")
            # if exit button is clicked leave current widget and break loop
            if stop.get() == 1:
                broken = 1
                break
    # if for loop is done (and was not interrupted with break), retrain model:
    if broken == 0:
        retrain.set(1)

    # destroy current widget:
    #for widget in root.winfo_children():
        #if isinstance(widget, tk.Toplevel):
            #widget.destroy()
    root.quit()
    root.destroy()

    retrain_int = retrain.get()
    return files_list_N, files_list_P, retrain_int



# %% Load Online Learning method
def online_learner(
    train_path, test_path,
    holdout_path, pt_model_path="", pt_model=None,
    trained_embedder=None, threshold=0.01,
    batch_size=32, store=True, epochs=20, dropout=0.5):
    '''
    INPUT: Pretrained Model (AE) or Trained Model (Classifier), Trainset, Testset,
    Holdoutset (path to folders containing N and P)
    Threshold for using as additional data 
    Store to decide if resulting model should be stored

    OUTPUT: Holdout performance (f1) without online learning/with online learning
    and delta of both settings
    '''

    # Instantiate new Embedder modeler
    embedder = EmbedderModeler(train_path, batch_size=batch_size,
                               size_dense=32, dropout=dropout,
                               epochs=epochs, dim=200)
    # Create new data generator for yielding images:
    datagen = embedder.generator()
    #datagen_aug = embedder.generator(horizontal_flip=True, rotation=180)
    # Define callbacks
    callback_list_classifier = embedder.callbacks(
        NAME="best_model_Classifier",
        mc_monitor="val_acc"
    )
    # If there is only a pretrained (AE) model:
    # Train pretrained model on trainset:
    if trained_embedder is None:
        print("Train on pretrained model!")
        if pt_model is None:
            print("LOAD PRETRAINED MODEL!")
            # load pretrained model:
            pretrained_model = load_model(pt_model_path)

        # Train on pretrained model and return best classifier (unsliced):
        trained_embedder = embedder.train_on_pretrained(datagen,
                                                        pt_model_path,
                                                        callback_list_classifier,
                                                        pt_model)

        # Slice last 4 layers:
        # trained_embedder = trained_embedder.layers[:-4]
        trained_embedder.pop()
        trained_embedder.pop()
        trained_embedder.pop()
        # trained_embedder.pop()
        trained_embedder.summary()

    else:
        # if given as method input slice dense classifer
        # trained_embedder = tf.keras.Sequential(trained_embedder.layers[:-4])
        #print("summary before pop: ")
        # trained_embedder.summary()
        trained_embedder.pop()
        trained_embedder.pop()
        trained_embedder.pop()
        # trained_embedder.pop()
        #print("summary after pop: ")
        trained_embedder.summary()

    print("Classifier MODEL TRAINED!")
    # Predict Embeddings of trainset with trained model:
    # define generator for prediction
    pred_generator = embedder.generator(val_split=0.0)

    # Generator for pred of train data
    pred_gen_train = pred_generator.flow_from_directory(train_path,
                                                        target_size=(300, 300),
                                                        color_mode="grayscale",
                                                        batch_size=1,
                                                        shuffle=False,
                                                        class_mode="binary",
                                                        seed=42)

    #print("Does it work to train sliced model without compiling?")
    #trained_embedder.fit(pred_gen_train, steps_per_epoch=int(pred_gen_train.samples), epochs=embedder.epochs)

    # number of files in train dir:
    nr_train = len(pred_gen_train.filenames)
    # Predict Embeddings of holdoutset with trained model:
    emb_train = trained_embedder.predict_generator(pred_gen_train,
                                                   steps=nr_train)

    # print(emb_train)
    print(emb_train.shape)
    #emb_train = np.reshape(emb_train,(nr_train,embedder.dim))
    print("TRAIN EMBEDDINGS EXTRACTED!")
    # store train_labels:
    count = 0
    labels_train = []
    for _, labels in pred_gen_train:
        labels_train.append(labels[0])
        count += 1
        if count == nr_train:
            break

    print("TRAIN LABELS STORED!")
    # Generator for pred of holdout data
    pred_gen_ho = pred_generator.flow_from_directory(holdout_path,
                                                     target_size=(300, 300),
                                                     color_mode="grayscale",
                                                     batch_size=1,
                                                     shuffle=False,
                                                     class_mode='binary',
                                                     seed=42)

    # number of files in holdout dir:
    nr_ho = len(pred_gen_ho.filenames)
    print(str(nr_ho), "ho files!")
    # Predict Embeddings of holdoutset with trained model:
    emb_ho = trained_embedder.predict_generator(pred_gen_ho, steps=nr_ho)
    #print("emb_ho: ", emb_ho)
    print("HO EMB EXTRACTED!")
    # store holdout_labels:
    labels_ho = []
    count = 0
    for _, labels in pred_gen_ho:
        labels_ho.append(labels[0])
        count += 1
        if count == nr_ho:
            break

    # Generator for pred of test data
    pred_gen_test = pred_generator.flow_from_directory(test_path,
                                                       target_size=(300, 300),
                                                       color_mode="grayscale",
                                                       batch_size=1,
                                                       shuffle=False,
                                                       class_mode='binary',
                                                       seed=42)
    # number of files in test dir:
    nr_test = len(pred_gen_test.filenames)
    print(str(nr_test), "test files!")
    # Predict Embeddings of testset with trained model:
    emb_test = trained_embedder.predict_generator(pred_gen_test, steps=nr_test)
    # store test_labels:
    labels_test = []
    count = 0
    for _, labels in pred_gen_test:
        labels_test.append(labels[0])
        count += 1
        if count == nr_test:
            break

    print("TESTEMBEDDINGS EXTRACTED!")
    # Store filenames (as list) of pred_gen_test
    # in order it is yielding the images of test_dir:
    filenames_test = pred_gen_test.filenames
    #print("filenames_test: ", filenames_test)
    # Train scaler on trainset:
    scaler = StandardScaler()
    emb_train_scaled = scaler.fit_transform(emb_train)
    print("emb_train: shape ", emb_train.shape)
    print("emb_test: shape ", emb_test.shape)
    # Transform test and holdout set:
    emb_test_scaled = scaler.transform(emb_test)
    print("emb_test_scaled: shape ", emb_test_scaled.shape)
    emb_ho_scaled = scaler.transform(emb_ho)
    print("ALL EMBEDDINGS TRANSFORMED!")
    # Train MLP-Classifier with scaled train embeddings:
    mlp_1 = MLPClassifier(hidden_layer_sizes=(64, 32),
                          solver="adam", batch_size=32)
    mlp_1.fit(emb_train_scaled, labels_train)
    print("MLP1 FIT!")
    # Predict classes of holdoutset:
    pred_ho_1 = mlp_1.predict(emb_ho_scaled)

    # Predict class probability of testset
    pred_prob_test = mlp_1.predict_proba(emb_test_scaled)
    print("pred_prob_test shape: ", pred_prob_test.shape)
    pred_prob_test = list(pred_prob_test)

    # add prob of image to be not a pitting:
    pred_prob_test = [element[0] for element in pred_prob_test]

    # For all predictions between thresholds in testset store indices in list:
    indifferent_indices = [
        pred_prob_test.index(i) for i in pred_prob_test
        if ((i > threshold) and (i < 1-threshold))
    ]

    # Store all corresponding filenames of indifferent images:
    indifferent_filenames = [filenames_test[idx]
                             for idx in indifferent_indices]
    # Add whole path to filenames (in testdata):
    test_path = test_path
    indifferent_filenames = [test_path +
                             "\\" + i for i in indifferent_filenames]
    #print("indifferent filenames: ", indifferent_filenames)
    # Store all corresponding labels of indifferent images (of testset):
    indifferent_labels = [labels_test[idx] for idx in indifferent_indices]
    # Determine number of indifferent images:
    nr_ind = len(indifferent_filenames)
    print("len ind_lables: ", str(len(indifferent_labels)),
          "len indifferent filenames: ", str(nr_ind))
    # Create new dir for indifferent images
    try:
        os.mkdir(r"data_bsd/ind_img")
        os.mkdir(r"data_bsd/ind_img\N")
        os.mkdir(r"data_bsd/ind_img\P")

    except:
        print("ind_img already exists!")

    # Copy all indifferent images to new dir:
    destination_path = r"data_bsd/ind_img"
    # Copy all indifferent images to new dir:
    matchcount = 0
    #print("ind_filenames", indifferent_filenames)
    # print("SOURCES!")
    # GUI module
    retrain = 0
    ind_fn_N, ind_fn_P, retrain = displayer(indifferent_filenames)
    print("returened list N:", ind_fn_N)
    print("retrain:", retrain, type(retrain))
    # Copy all indifferent images to new dir:
    for folder in os.listdir(test_path):
        folder_path = os.path.join(test_path, folder)
        for img in os.listdir(folder_path):
            source = os.path.join(folder_path, img)
            # print(source)
            destination_N = destination_path + '\\N\\' + img
            destination_P = destination_path + '\\P\\' + img
            if source in ind_fn_N:
                matchcount += 1
                shutil.copy(source, destination_N)
            if source in ind_fn_P:
                matchcount += 1
                shutil.copy(source, destination_P)

    print(str(matchcount), "INDIFFERENT FILES STORED!")
    #retrain = 1
    # If Model should be retrained:
    if retrain:
        ind_img_path = r"data_bsd/ind_img"
        gen_ind = pred_generator.flow_from_directory(ind_img_path,
                                                     target_size=(300, 300),
                                                     color_mode="grayscale",
                                                     batch_size=1,
                                                     class_mode='binary',
                                                     seed=42)

        # get indifferent labels:
        indifferent_labels = []
        count = 0
        for _, labels in gen_ind:
            indifferent_labels.append(labels[0])
            count += 1
            if count == matchcount:
                break

        #print("trained_embedder just before retraining ")
        # trained_embedder.summary()
        #trained_embedder.add(Dense(32, activation='relu'))
        #trained_embedder.add(Dense(32, activation='relu'))
        #trained_embedder.add(Dense(1, activation='sigmoid'))
        # Train trained_model with indifferent images anew:
        trained_embedder.fit(
            gen_ind,
            steps_per_epoch=int(gen_ind.samples),
            epochs=1,
            callbacks=callback_list_classifier
        )

        # Slice dense classifier
        # trained_embedder.pop()
        # trained_embedder.pop()
        # trained_embedder.pop()

        # Predict with trained_model_2 Embeddings of trainset
        emb_train_2 = trained_embedder.predict_generator(pred_gen_train,
                                                         steps=nr_train)
        print("dim of emb_train_2: ", str(emb_train_2.shape))
        # Predict with trained_model_2 Embeddings of indifferent testset
        emb_ind = trained_embedder.predict_generator(gen_ind,
                                                     steps=nr_ind)
        # Predict with trained_model_2 Embeddings of holdoutset
        emb_ho_2 = trained_embedder.predict_generator(pred_gen_ho, steps=nr_ho)
        print("dim of emb_ho_2: ", str(emb_ho_2.shape))
        # Train scaler on  new trainset:
        scaler_2 = StandardScaler()
        emb_train_2_scaled = scaler_2.fit_transform(emb_train_2)

        # Transform new holdout set and indifferent embeddings:
        emb_ind_scaled = scaler_2.transform(emb_ind)
        emb_ho_2_scaled = scaler_2.transform(emb_ho_2)

        # Train MLP anew with embeddings of trainset and indifferent testset
        mlp_2 = MLPClassifier(hidden_layer_sizes=(64, 32),
                              solver="adam", batch_size=32, warm_start=True)
        mlp_2.fit(emb_train_2_scaled, labels_train)
        print("Indifferent labels: ", indifferent_labels)
        mlp_2.fit(emb_ind_scaled, indifferent_labels)
        print("MLP2 FIT!")
        # Predict class for new holdoutset embeddings with MLP_2
        pred_ho_2 = mlp_2.predict(emb_ho_2_scaled)

        # Return classification report for both MLPs
        # Classification report for holdoutset without new training:
        #print(classification_report(labels_ho, pred_ho_1, digits=4))
        class_dict_1 = classification_report(labels_ho, pred_ho_1,
                                             digits=4, output_dict=True)
        # Classification report for holdoutset with new training:
        #print(classification_report(labels_ho, pred_ho_2, digits=4))
        class_dict_2 = classification_report(labels_ho, pred_ho_2,
                                             digits=4, output_dict=True)
        # Return deltas of f1 measure:
        diff_f1 = (class_dict_2["weighted avg"]["f1-score"]
                   - class_dict_1["weighted avg"]["f1-score"])
        diff_f2 = (class_dict_2["accuracy"]
                   - class_dict_1["accuracy"])
        print("Weighted F1-measure before retraining: ",
              class_dict_1["weighted avg"]["f1-score"])
        print("Weighted F1-measure after retraining: ",
              class_dict_2["weighted avg"]["f1-score"])
        print("Holdout accuracy before retraining: ",
              class_dict_1["accuracy"])
        print("Holdout accuracy after retraining: ",
              class_dict_2["accuracy"])
        print("Retraining of models with indifferent images improves weighted f1-measure by ", diff_f1, " !")
        print("Retraining of models with indifferent images improves accuracy by ", diff_f2, " !")

    # if retraining not wished end method:
    else:
        print("EXIT")
