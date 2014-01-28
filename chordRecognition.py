#!/opt/local/bin/python2.7

import mir_eval
import librosa
import pickle
import os
import random
import numpy as np
import matplotlib.pyplot as plt

"""
Jeff Scott and Alex Cannon 01/26/2014
HAMR at Columbia University

http://labrosa.ee.columbia.edu/hamr2014/proceedings/doku.php?id=simple_chord_recognition

"""



def computeAccuracy(predicted, actual):
    '''
    Returns the similarity between two arrays as the percentage of elements that match
    (order matters). e.g. computeAccuracy([1,2,3,4],[1,2,3,5]) would return 0.75.
    '''
    
    # Sometimes there are more frames in one vector than the other, just truncate to shortest length
    if len(predicted) < len(actual):
        actual = actual[:len(predicted)]
    if len(actual) < len(predicted):
        predicted = predicted[:len(actual)]        
    
    return len([0 for i, j in zip(predicted, actual) if i==j])/float(len(predicted))


def getChordTemplates(hop_length, sample_rate, cqtData, labels):

    """
    Computes the chord templates by taking the median of the spectrogram between the timestamps
    specified by the chord labels
    """

    # Parse individual label types from 'labels'
    times = labels[0]
    startTimes = [time[0] for time in times]
    chordLabels = labels[1]

    # Start/End in frame numbers, with error checking to ensure frame index doesn't
    # exceed data length
    startTimeFrames = [np.floor((startTime*sample_rate)/hop_length) \
                                        for startTime in startTimes \
                                        if np.floor((startTime*sample_rate)/hop_length) < cqtData.shape[1]]

    # Compute chord template as median of spectrum between chord labels                                    
    chordMedians = librosa.feature.sync(cqtData, map(int, startTimeFrames), aggregate=np.median)
    
    # Plot some chord templates
    if(0):

        # Plot sections of chord templates
        librosa.display.specshow(librosa.logamplitude(chordMedians[:,30:40]))
        plt.xticks(np.arange(10), np.asarray(chordLabels[30:40]))
      
        # Get center freqs of CQT
        numFreqs, numFrames = chordMedians.shape
        centerFreqs = np.logspace(np.log10(8.1757), np.log10(12543.85), numFreqs, endpoint=False)
        ticks = np.arange(numFreqs)
        
        plt.yticks(ticks[::10], np.round(centerFreqs[::10]).astype(int))
        plt.xlabel('Chord')
        plt.ylabel('Frequency')
        plt.show()

    return chordMedians

def estimateChords(chordMedians, testCQT):
    '''
    Returns a list of detected chords given song spectrogram data
    '''
    chordEstimates = np.dot(np.power(chordMedians.T,2),np.power(testCQT,2))
    
    return chordEstimates


def main():
    
    # Min/Max Frequency bin indicies of the CQT used in the templates
    cqtStartFreqIdx = 15
    cqtEndFreqIdx = 65

    # If this is too large it takes forever to run
    numTrainingSongs = 40  
    
    '''
    Data is comprised of two files for every song:
        (1) A pickled python object with the name analysis_<song_id>.pickle
        (2) Chord labelings for that song with the name labels_<song_id>.txt
                that store the start time, end time, and chord label on each line
    '''

    # get song ids
    trainingSongIDs = []
    dataFiles = os.listdir("./data")
    for file in dataFiles:
        if file.endswith(".pickle"):
            trainingSongIDs.append(file.split("analysis_")[1].split(".pickle")[0])  

    testSongIDs = trainingSongIDs[-20:]
    trainingSongIDs = trainingSongIDs[:-20] 
    
    # reserve 20% of those song ids for testing
    # TODO: Uncomment this to randomly select testing vs. training data

    testSongIDs = []
    numTestSongs = (len(trainingSongIDs)/10) * 2
    for i in range(numTestSongs):
        testSongIDs.append(trainingSongIDs.pop(random.randrange(len(trainingSongIDs))))


    ####################################################
    #
    # COMPUTING THE CHORD TEMPLATES
    #
    ####################################################  
    numIter = 0    
    labelDictionary = []
    for i, songID in enumerate(trainingSongIDs[:numTrainingSongs]):

        print "Training on song " + songID + "..."

        # Load up the data
        trainingSong = pickle.load(open("data/analysis_" + songID + ".pickle", 'rb'))
        trainingLabels = mir_eval.io.load_annotation("data/labels_" + songID + ".txt")
        labelDictionary = labelDictionary + trainingLabels[1]
        
        params = trainingSong['PARAMETERS']

        # each row in templates is a chord, where templates[i] is a chord of type labels[i]
        # TODO: aggregate labels from each iteration into a "master labels" numpy array, such
        # that the above condition is satisfied
        
        # Get parameters used to compute CQT
        hop_length = params['stft']['hop_length']
        cqtData = trainingSong['cqt']
        cqtData = cqtData[cqtStartFreqIdx:cqtEndFreqIdx,:]
        sample_rate = params['load']['sr']
        
        
        # Compute the templates
        if i == 0:
            templates = getChordTemplates(hop_length, sample_rate, cqtData, trainingLabels)
        else:
            templates = np.concatenate((templates, getChordTemplates(hop_length, sample_rate, cqtData, trainingLabels)), axis=1)
        numIter+=1

    print 'Chord templates computed!\n\n'
    # This mod operation collapses all the numeric labels into [1, 12] - root estimation only
    # Other operations can be performed to yield maj/min and maj/min with 7ths
    labelDictionary = [int(x) for x in labelDictionary]
    labelDictionary = [x % 12 for x in labelDictionary]

    # Plot the giant matrix of chord templates
    if(0):
        librosa.display.specshow(librosa.logamplitude(templates))
        plt.title('CQT Chord Templates')
        plt.show()


    ####################################################
    #
    # TEST CHORD TEMPLATES
    #
    ####################################################

    accuracies = []
    for songID in testSongIDs:
        print "Testing on song " + songID + "...",

        # Load up the data
        testSong = pickle.load(open("data/analysis_" + songID + ".pickle", 'rb'))
        testLabels = mir_eval.io.load_annotation("data/labels_" + songID + ".txt")
        groundTruthlabels = testLabels[1]
        
        # get estimated chords and compare to actual chords, calculate accuracy
        cqtData = testSong['cqt']
        cqtData = cqtData[cqtStartFreqIdx:cqtEndFreqIdx,:]
        chordEstimates = estimateChords(templates, cqtData)        
        
        # Find the indicies of the maximum value in the estimates matrix then grab the associated chord label
        chordDecisionIdx = chordEstimates.argmax(axis=0)        
        predictedChords = [labelDictionary[n] for n in chordDecisionIdx]

        #Create ground truth chord label array of same size as prediction        
        times = testLabels[0]
        startTimes = [time[0] for time in times]        
        params = trainingSong['PARAMETERS']
        hop_length = params['stft']['hop_length']
        sample_rate = params['load']['sr']
            
        # Get chord labels in terms of frame number
        startTimeFrames = [np.floor((startTime*sample_rate)/hop_length) for startTime in startTimes]
        startTimeFrames = startTimeFrames + np.floor((times[1][-1]*sample_rate)/hop_length)
                
        # Repeat chord label for desired number of times and concatenate
        groundTruthChords = []
        for g in np.arange(len(groundTruthlabels)-1):
            groundTruthChords = groundTruthChords + [groundTruthlabels[g]]*(startTimeFrames[g+1] - startTimeFrames[g])
            

        # Do the same thing as in computing the templates (root only) - this should really be a function
        groundTruthChords = [int(x) for x in groundTruthChords]
        predictedChords = [x % 12 for x in predictedChords]

        # Compute the accuracy and output
        acc = computeAccuracy(groundTruthChords, predictedChords)
        print 'Song accuracy: ' + "%.3f" % acc
        accuracies.append(acc)        
    
        numIter+=1

    # Final output
    avgacc = sum(accuracies)/float(len(accuracies))
    print "Accuracy: " + "%.4f" % (avgacc * 100) + "%"
    print 'There were ' + str(numIter) + ' TestSongs'

main()
