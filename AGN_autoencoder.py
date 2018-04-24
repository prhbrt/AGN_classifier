import os, sys, glob
import pandas as pd
import numpy as np
#import skimage.io
import pickle
import time
import itertools
from textwrap import wrap
import multiprocessing

#ML libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import scipy.stats as stats
from sklearn import manifold
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE #single core TSNE, sklearn.
from MulticoreTSNE import MulticoreTSNE as multiTSNE #multicore TSNE, not sklearn implementation.

import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set()
# sns.set_style({'legend.frameon': True})

from matplotlib import rcParams
rcParams['font.family'] = 'Latin Modern Roman'

import matplotlib.colors as colors
import matplotlib.cm as cmx

figloc = 'Figures/'


#functions for saving/loading objects (arrays, data frames, etc)
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

#function to create a plot of confusion matrix after a classifier has been run
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Showing normalized confusion matrix")
    else:
        print('Showing confusion matrix, without normalization')
    #print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#Function to prepare data for Machine Learning
def prepare_data(datasource = 'SDSS'):

    try:
        print('Trying to load old data')
        Xtrain = np.load(f'{datasource}_Xtrain.npy')
        Xtest = np.load(f'{datasource}_Xtest.npy')
        Ytrain = np.load(f'{datasource}_Ytrain.npy')
        Ytest = np.load(f'{datasource}_Ytest.npy')
    except:
        if datasource == 'SDSS':
            if verbose==True: print('loading saved tables from disk: '+filename)
            data_table=load_obj(filename)
            #if verbose==True: print(data_table)
            if verbose==True: print('The table loaded is of shape: {0}'.format(data_table.shape))
            #trim away unwanted columns
            #data_table_trim=data_table.drop(columns=['#ra', 'dec', 'z', 'class'])
            data_table_trim=data_table.drop(columns=trim_columns)
            all_features=data_table_trim[:]
            #print(all_features)
            all_classes=data_table['class']
            #split data up into test/train
            features_train, features_test, classes_train, classes_test = train_test_split(all_features, all_classes, train_size=train_percent, random_state=0, stratify=all_classes)
            class_names=np.unique(all_classes)
            feature_names=list(all_features)
            if verbose==True: print('feature names are: ', str(feature_names))
            #return dictionary: features_train, features_test, classes_train, classes_test, class_names, feature_names

            # Xtrain_in = data['features_train']
            # Ytrain_in = data['classes_train']
            # Xtest_in = data['features_test']
            # Ytest_in = data['classes_test']

            print('Obtaining data arrays...')
            #obtain data arrays
            # X = np.empty((0, Xtrain_in.shape[1]))
            X = []
            for k in features_train.keys():
                print(f'Adding {k}')
                X.append(np.concatenate((features_train[k], features_test[k])))

            Y = []
            for k in classes_train.keys():
                Y.append(classes_train[k])
            for k in classes_test.keys():
                Y.append(classes_test[k])

            X = np.array(X).T
            Y = np.array(Y)

            ##################
            # Remove rows with missing values or radio columns
            ##################
            '''
            print('Removing missing values')
            #remove all rows which contain a magnitude 0
            removeloc = ~np.any(X == 0, axis=1)
            X = X[removeloc]
            Y = Y[removeloc]
            '''
            #remove the last 5 columns of radio data
            X = X[:,:-5]

            #remove all rows which contain a too small magnitude
            removeloc = ~np.any(X < -10, axis=1)
            X = X[removeloc]
            Y = Y[removeloc]

            #slice train and test data
            Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.001, random_state=42) 
        elif datasource == 'ZFOURGE':
            dropcolumns = np.array(['use', 'star', 'lssfr', 'Unnamed: 0', 'ir_agn', 'radio_agn', 'xray_agn'])

            data = pd.read_csv('ZFOURGE_Data.csv')
            data = data.drop(dropcolumns, 1)

            #filter out extreme outliers like 10^24 mag
            for jj in data.columns:
                if jj.startswith('f_'):
                    data = data[data[jj] < np.percentile(data[jj], 99.9)]

            Y = np.array(data['classes'])
            X = np.array(data.drop('classes', 1))

            print(X.shape)
            print(Y.shape)

            #set all AGNs to 1
            Y[Y > 0] = 1

            #slice train and test data
            Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.001, random_state=42) 

            feature_names = data.columns

        np.save(f'{datasource}_Xtrain.npy', Xtrain)
        np.save(f'{datasource}_Xtest.npy', Xtest)
        np.save(f'{datasource}_Ytrain.npy', Ytrain)
        np.save(f'{datasource}_Ytest.npy', Ytest)

        print(feature_names)

    return Xtrain, Xtest, Ytrain, Ytest

#Function to create a TSNE plot
def TSNE_plot(all_features, all_classes, n_iter=2000, lrate=500, verbose=False, multicore=False):
    if multicore==False:
        print('applying TSNE...')
        tsne = TSNE(n_components=2, n_iter=n_iter, learning_rate=lrate, verbose=verbose)
    if multicore==True:
        print('applying multicore TSNE...')
        tsne = multiTSNE(n_components=2, n_jobs=-1, n_iter=n_iter, learning_rate=lrate, verbose=verbose)
    reduced_data=tsne.fit_transform(all_features)
    #make plot
    cols = {"GALAXY": "blue", "STAR": "green", "QSO": "red"}
    #plt.scatter(reduced_data[:,0], reduced_data[:,1], c=data_table['peak'][:])
    names = set(all_classes)
    x,y = reduced_data[:,0], reduced_data[:,1]
    for name in names:
        cond = all_classes == name
        plt.plot(x[cond], y[cond], linestyle='none', marker='o', label=name)
    plt.legend(numpoints=1)
    plt.savefig('tSNE_classes.png')
    plt.show()

#Function to run randon forest pipeline with feature pruning and analysis
def RF_pipeline(data, train_percent, n_jobs=-1, n_estimators=500, pruning=False):
    rfc=RandomForestClassifier(n_jobs=n_jobs,n_estimators=n_estimators,random_state=2,class_weight='balanced')
    pipeline = Pipeline([ ('classification', RandomForestClassifier(n_jobs=n_jobs, n_estimators=n_estimators,random_state=2,class_weight='balanced')) ])
    #do the fit and feature selection
    pipeline.fit(data['features_train'], data['classes_train'])
    # check accuracy and other metrics:
    classes_pred = pipeline.predict(data['features_test'])
    accuracy_before=(accuracy_score(data['classes_test'], classes_pred))
    print('hello')
    report=classification_report(data['classes_test'], classes_pred, target_names=np.unique(data['class_names']))
    print('accuracy before pruning features: {0:.2f}'.format(accuracy_before))
    #print('We should check other metrics for a full picture of this model:')
    print('--'*30+'\n Random Forest report before feature pruning:\n',report,'--'*30)

    #make plot of feature importances
    clf=[]
    clf=pipeline.steps[0][1] #get classifier used. zero because only 1 step.
    importances = pipeline.steps[0][1].feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_],
             axis=0)
    indices = np.argsort(importances)[::-1]
    feature_names_importanceorder=[]
    for f in range(len(indices)):
        #print("%d. feature %d (%f) {0}" % (f + 1, indices[f], importances[indices[f]]), feature_names[indices[f]])
        feature_names_importanceorder.append(str(data['feature_names'][indices[f]]))
    plt.figure()
    plt.title("\n".join(wrap("Feature importances. n_est={0}. Trained on {1}% of data. Accuracy before={2:.3f}".format(n_estimators,train_percent*100,accuracy_before))))
    plt.bar(range(len(indices)), importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(len(indices)), indices)
    plt.xlim([-1, len(indices)])
    plt.xticks(range(len(indices)), feature_names_importanceorder, rotation='vertical')
    plt.tight_layout()
    plt.savefig('Feature_importances.png')
    #plt.show()

    #normal scatter plot for one class
    #plt.scatter(reduced_data[:,0], reduced_data[:,1], c=list(map(cols.get, data_table['class'][:])), label=set(data_table['class'][:]) )
    #plt.colorbar(label='Peak Flux, Jy')
    #plt.show()
    classes_important_pred=[]
    if pruning==True:
        #first choose a model to prune features, then put it in pipeline - there are many we could try
        #lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(features_train, artists_train)
        rfc=RandomForestClassifier(n_jobs=n_jobs,n_estimators=n_estimators,random_state=2,class_weight='balanced')
        modelselect='rfc' #set accordingly
        pipeline_prune = Pipeline([
            ('feature_selection', SelectFromModel(rfc)),
            ('classification', RandomForestClassifier(n_jobs=-1, n_estimators=n_estimators,random_state=2,class_weight='balanced'))
        ])
        pipeline_prune.fit(data['features_train'], data['classes_train']) #do the fit and feature selection
        classes_important_pred = pipeline_prune.predict(data['features_test'])
        accuracy_after=(accuracy_score(data['classes_test'], classes_important_pred))
        #print('accuracy before pruning features: {0:.2f}'.format(accuracy_before))
        print('Accuracy after pruning features: {0:.2f}'.format(accuracy_after))
        print('--'*30)
        print('Random Forest report after feature pruning:')
        print(classification_report(data['classes_test'], classes_important_pred, target_names=data['class_names']))
        print('--'*30)

        #make plot of feature importances
        clf=[]
        clf=pipeline_prune.steps[1][1] #get classifier used
        importances = pipeline_prune.steps[1][1].feature_importances_
        std = np.std([tree.feature_importances_ for tree in clf.estimators_],
                     axis=0)
        indices = np.argsort(importances)[::-1]
        # Now we've pruned bad features, create new feature_names_importanceorder_pruned array
        # Print the feature ranking to terminal if you want, but graph is nicer
        #print("Feature ranking:")
        feature_names_importanceorder_pruned=[]
        for f in range(len(indices)):
            #print("%d. feature %d (%f) {0}" % (f + 1, indices[f], importances[indices[f]]), feature_names[indices[f]])
            feature_names_importanceorder_pruned.append(str(data['feature_names'][indices[f]]))
        # Plot the feature importances of the forest
        plt.figure()
        try:
            plt.title("\n".join(wrap("Feature importances pruned with {0}. n_est={1}. Trained on {2}% of data. Accuracy before={3:.3f}, accuracy after={4:.3f}".format(modelselect,n_estimators,train_percent*100,accuracy_before,accuracy_after))))
        except: #having issues with a fancy title? sometimes too long?
            plt.title('After pruning features:')
        plt.bar(range(len(indices)), importances[indices],
               color="r", yerr=std[indices], align="center")
        plt.xticks(range(len(indices)), indices)
        plt.xlim([-1, len(indices)])
        plt.xticks(range(len(indices)), feature_names_importanceorder_pruned, rotation='vertical')
        plt.tight_layout()
        plt.savefig('Feature_importances_pruned.png')
        #plt.show()

    return classes_pred, classes_important_pred, clf
    #if not None:
    #    return classes_important_pred

def savemodel(model, savename):
    """
    Save a Keras model
    """
    model_json = model.to_json()
    with open(f'{savename}.json', 'w') as json_file:
        json_file.write(model_json)
    #save weights
    model.save_weights(f'{savename}.h5')
    print('Saved NN model to disk')

def loadmodel(savename):
    """
    Load a Keras model from disk. You have to recompile the model
    after loading it.
    """
    from keras.models import model_from_json

    json_file = open(f'{savename}.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    # load weights into new model
    model.load_weights(f'{savename}.h5')
    print('Loaded model from disk')

    return model

def autoencoder(Xtrain, Xtest, Ytrain, Ytest, datasource = 'SDSS'):
    """
    Apply autoencoder to SDSS data
    """
    from keras.layers import Input, Dense, Dropout
    from keras.models import Model
    from keras.layers.normalization import BatchNormalization
    from keras import optimizers

    from sklearn.svm import SVC
    from sklearn import metrics

    import hdbscan

    #whether to make a new model
    newmodel = False

    #version name of the model
    modelname = f'_v23'

    #location of the to be saved models
    modelloc = 'Models_autoencoder/'

    #whether to run a clustering algorithm
    runclustering = False

    #whether to plot the train or test set
    plottest = False

    n_epochs = 20
    batch_size = 16
    lossfunction = 'mean_squared_error'
    optimizer = 'Adam'
    learning_rate = 1e-4
    mid_activation = 'relu'
    encoder_activation = 'relu'
    batchnormalization = True
    dropout = True

    print(f'Train size: {Xtrain.shape}')
    print(f'Test size: {Xtest.shape}')

    if optimizer == 'Adam':
        optim = optimizers.Adam(lr = learning_rate)
    elif optimizer == 'rmsprop':
        optim = optimizers.RMSprop(lr = learning_rate)
    
    if newmodel:
        inputs = Input(shape=(Xtrain.shape[1],))
        if batchnormalization:
            x = BatchNormalization()(inputs)
            x = Dense(60, activation = mid_activation)(x)
        else:
            x = Dense(60, activation = mid_activation)(inputs)
        if dropout:
            x = Dropout(0.3)(x)
        x = Dense(40, activation = mid_activation)(x)
        if dropout:
            x = Dropout(0.3)(x)
        x = Dense(12, activation = mid_activation)(x)
        encoded = Dense(2, activation = encoder_activation)(x)

        x = Dense(12, activation = mid_activation)(encoded)
        x = Dense(40, activation = mid_activation)(x)
        if dropout:
            x = Dropout(0.3)(x)
        x = Dense(60, activation = mid_activation)(x)
        if dropout:
            x = Dropout(0.3)(x)
        decoded = Dense(Xtrain.shape[1], activation = 'relu')(x)
        # decoded = LeakyReLU(alpha=.001)(x)

        encoder = Model(inputs, encoded)
        autoencoder = Model(inputs, decoded)

        autoencoder.compile(loss=lossfunction,
                          optimizer=optim,
                          metrics=['acc'])

        autoencoder.summary()

        history = autoencoder.fit(Xtrain, Xtrain, 
                        epochs = n_epochs, 
                        batch_size = batch_size,
                        shuffle = True,
                        validation_split = 0.05)

        savemodel(autoencoder, f'{modelloc}autoencoder{modelname}')
        savemodel(encoder, f'{modelloc}encoder{modelname}')

        #plot the progression of the loss and the accuracy
        plt.plot(history.history['loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(f'{figloc}Model_history_loss{modelname}.png', dpi = 300, bbox_inches = 'tight')
        plt.show()

        plt.plot(history.history['acc'])
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.savefig(f'{figloc}Model_history_accuracy{modelname}.png', dpi = 300, bbox_inches = 'tight')
        plt.show()

        #save the autoencoder settings
        modelsettings = pd.read_csv('modelsettings.csv')

        appenddata = np.array([
                                modelname[1:],
                                n_epochs,
                                batch_size,
                                lossfunction,
                                optimizer,
                                encoder_activation,
                                mid_activation,
                                learning_rate,
                                batchnormalization,
                                dropout,
                                datasource
                                ]).T

        if modelname[1:] in modelsettings['Version']:
            appendloc = np.where(modelname[1:] == modelsettings['Version'])
        else:
            appendloc = len(modelsettings)
        modelsettings.loc[appendloc] = appenddata

        modelsettings.to_csv('modelsettings.csv', index = False)
        
        del autoencoder, encoder

    autoencoder = loadmodel(f'{modelloc}autoencoder{modelname}')
    encoder = loadmodel(f'{modelloc}encoder{modelname}')

    encoder.compile(loss=lossfunction,
                          optimizer=optim,
                          metrics=['acc'])

    if not runclustering:
        if plottest:
            Yenc = encoder.predict(Xtest)
            Xplot = Xtest
            Yplot = Ytest
            unique_labels = np.unique(Ytest)
            savename = f'{figloc}{datasource}_autoencoder{modelname}_test.png'
        else:
            Yenc = encoder.predict(Xtrain)
            Xplot = Xtrain
            Yplot = Ytrain
            savename = f'{figloc}{datasource}_autoencoder{modelname}_train.png'

        #find the unique labels 
        unique_labels = np.unique(Ytrain)

        #obtain colours for the different labels
        jet = plt.get_cmap('summer') 
        cNorm  = colors.Normalize(vmin = 0, vmax = len(unique_labels) - 1)
        scalarMap = cmx.ScalarMappable(norm = cNorm, cmap = jet)

        if len(unique_labels) < 3:
            colours = ['#09A309', '#E3562B', '#773BE3']
        else:
            colours = scalarMap.to_rgba

        stepsize = 50
        #plot in batches of stepsize to prevent the overlap of scatter points
        for j in range(0, Yenc.shape[0] - stepsize, stepsize):
            print(f'{j}/{Yenc.shape[0]}')
            for i, label in enumerate(unique_labels):
                plt.scatter(Yenc[j:j + stepsize][Yplot[j:j + stepsize] == label, 0], Yenc[j:j + stepsize][Yplot[j:j + stepsize] == label, 1], s = 2, color = colours[i])

        #turn of axis labels
        # ax = plt.gca()
        # ax.xaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
        # ax.yaxis.set_major_formatter(matplotlib.ticker.NullFormatter())

        #make the labels for the legend
        labels = []
        for lab in unique_labels:
            labels.append(f'{lab} ({np.sum(Yplot == lab)})')

        plt.legend(labels, loc = 'best')
        plt.title('Autoencoder 2D encoder output')
        plt.savefig(savename, dpi = 300, bbox_inches = 'tight')
        plt.show()
    else:
        print('Running HDBSCAN...')
        # Xtrain_enc = encoder.predict(Xtrain)
        Xtest_enc = encoder.predict(Xtest)
        
        
        clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
        cluster_labels = clusterer.fit_predict(Xtest)

        unique_cluster_labels, cluster_counts = np.unique(cluster_labels, return_counts = True)
        unique_true_labels, true_count = np.unique(Ytest, return_counts = True)

                #convert the string labels to integers
        sortloc_cluster = np.argsort(cluster_counts)
        sortloc_true = np.argsort(true_count)
        unique_cluster_labels = unique_cluster_labels[sortloc_cluster]
        unique_true_labels = unique_true_labels[sortloc_true]

        Ytest_int = np.zeros(len(Ytest))
        for lab, i in zip(unique_true_labels, unique_cluster_labels):
            Ytest_int[Ytest == lab] = i

        ##### Quality measures #####
        confusion_matrix = metrics.confusion_matrix(Ytest_int, cluster_labels)
        print(confusion_matrix)

        print('Plotting confusion matrix...')
        plot_confusion_matrix(confusion_matrix, classes=unique_true_labels, title='Confusion matrix')
        plt.savefig(f'{figloc}SDSS_autoencoder{modelname}_HDBSCAN_confmat.png', dpi = 300, bbox_inches = 'tight')
        plt.close()

        # print('-'*40)
        # print(f'Train accuracy: {metrics.accuracy_score(Ytrain, Ytrain_pred)}')
        # print(f'Train f-score: {metrics.f1_score(Ytrain, Ytrain_pred)}')
        # print(f'Train recall: {metrics.recall_score(Ytrain, Ytrain_pred)}')
        # print('-'*40)
        print(f'Accuracy: {metrics.accuracy_score(Ytest_int, cluster_labels)}')
        # print(f'Test f-score: {metrics.f1_score(Ytest, Ytest_pred)}')
        # print(f'Recall: {metrics.recall_score(Ytest_int, cluster_labels)}')



        #obtain colours for the different labels
        jet = plt.get_cmap('plasma') 
        cNorm  = colors.Normalize(vmin = 0, vmax = len(unique_cluster_labels) - 1)
        scalarMap = cmx.ScalarMappable(norm = cNorm, cmap = jet)

        stepsize = 50
        for j in range(0, Xtest_enc.shape[0] - stepsize, stepsize):
            print(f'{j}/{Xtest_enc.shape[0]}')
            for i, label in enumerate(unique_cluster_labels):
                plt.scatter(Xtest_enc[j:j + stepsize][cluster_labels[j:j + stepsize] == label, 0], Xtest_enc[j:j + stepsize][cluster_labels[j:j + stepsize] == label, 1], s = 2, color = scalarMap.to_rgba(i))

        #make the labels for the legend
        labels = []
        for lab in unique_cluster_labels:
            labels.append(f'{lab} ({np.sum(cluster_labels == lab)})')

        plt.legend(labels, loc = 'best')
        plt.savefig(f'{figloc}SDSS_autoencoder{modelname}_HDBSCAN.png', dpi = 300, bbox_inches = 'tight')
        plt.show()

#########################################################################
########################## END OF FUNCTIONS #############################
#########################################################################
######################### DO MACHINE LEARNING ###########################
#########################################################################


#Define inputs
input_table='test_query_table_TGSSadded'
trim_columns=['#ra', 'dec', 'z', 'class'] #columns you don't want ML to use
#Classifier variables
train_percent=0.1 #fraction
n_estimators=100 #number of trees

#Load and prepare data for machine learning
Xtrain, Xtest, Ytrain, Ytest = prepare_data(datasource = 'ZFOURGE')
#Prepared_data is a dictionary with keys: features_train, features_test, classes_train, classes_test, class_names, feature_names
#Note that class_names are unique names


#run the autoencoder
autoencoder(Xtrain, Xtest, Ytrain, Ytest, datasource = 'ZFOURGE')





'''

#Run random forest classifier
rf_start_time=time.time() #note start time of RF
print('Starting random forest pipeline...')
classes_pred, classes_important_pred, clf = RF_pipeline(prepared_data, train_percent, n_jobs=-1, n_estimators=n_estimators, pruning=False)
rf_end_time=time.time()
print('Finished! Run time was: ', rf_end_time-rf_start_time)

#Create confusion matrix plots from RF classifier
cnf_matrix = confusion_matrix(prepared_data['classes_test'], classes_pred)
#np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=prepared_data['class_names'], title='Confusion matrix, without normalization')
# Plot normalized confusion matrix
#plt.figure()
#plot_confusion_matrix(cnf_matrix, classes=names, normalize=True,
#                      title='Normalized confusion matrix')
plt.savefig('Confusion_matrix.png')
plt.show()


#run tSNE and make plot (warning, takes 10 minutes for 10000 sources)
print('Running tSNE, note that this could take more than an hour if you have >1e5 sources... try turning on the multicore flag, but note that multicore TSNE is not the same algorithm as SKLearn.')
prepared_data = prepare_data(input_table, trim_columns, train_percent, verbose=True, tsne=True)
#tsne=True means don't split data into test/train
#print('you have {0} sources...'.format(len(prepared_data['all_features'])))
TSNE_plot(prepared_data['all_features'], prepared_data['all_classes'], n_iter=2000, lrate=500, verbose=False, multicore=False)
'''

#something else

