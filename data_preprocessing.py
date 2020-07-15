import os
import datetime
import time
import pandas as pd
import scipy
import glob
import pympi
import sklearn

import random as rn
rn.seed(42)

import numpy as np
np.random.seed(42)

def robot_is_speaking(eaf, dfdata, stepSize):
    """
        robot_is_speaking segment as binary:
            Robot is speaking = 1
            Robot is listening = 0
    """
    dfout = pd.DataFrame()
    t0 = dfdata.index[0]
    indexout = []
    RobotSpeak = []
    Robotdur = []
    userdur = []
    Robotdurt = []
    userdurt = []
    t = t0
    deb = sorted(eaf.get_annotation_data_for_tier('TurnTalk'))[0][0] + 1
    fin = sorted(eaf.get_annotation_data_for_tier('TurnTalk'))[0][1] - 1
    typ = sorted(eaf.get_annotation_data_for_tier('TurnTalk'))[0][2]
    
    for i in range(len(sorted(eaf.get_annotation_data_for_tier('Transcription')))):
        deb = sorted(eaf.get_annotation_data_for_tier('Transcription'))[i][0] + 1
        fin = sorted(eaf.get_annotation_data_for_tier('Transcription'))[i][1] - 1
        typ = eaf.get_annotation_data_at_time('TurnTalk', deb)[0][2]
        
        while t <= t0 + datetime.timedelta(milliseconds=fin):
            if t >= t0 + datetime.timedelta(milliseconds=deb):
                if typ == 'Robot':
                    indexout.append(t)
                    RobotSpeak.append(1)
                    Robotdur.append(float(fin - deb + 2) / 1000)
                    Robotdurt.append((t - datetime.timedelta(milliseconds=deb + 1) - t0).total_seconds())
                    userdur.append(0)
                    userdurt.append(0)
                elif typ == 'User':
                    indexout.append(t)
                    RobotSpeak.append(0)
                    Robotdur.append(0)
                    Robotdurt.append(0)
                    userdur.append(float(fin - deb + 2) / 1000)
                    userdurt.append((t - datetime.timedelta(milliseconds=deb + 1) - t0).total_seconds())
                else:
                    indexout.append(t)
                    RobotSpeak.append(np.nan)
                    Robotdur.append(np.nan)
                    userdur.append(np.nan)
                    Robotdurt.append(np.nan)
                    userdurt.append(np.nan)
            else:
                indexout.append(t)
                RobotSpeak.append(np.nan)
                Robotdur.append(np.nan)
                userdur.append(np.nan)
                Robotdurt.append(np.nan)
                userdurt.append(np.nan)
            t = t + pd.to_timedelta(stepSize, unit='ms')
    while t <= dfdata.index[-1]:
        if t >= t0 + datetime.timedelta(milliseconds=deb) and t >= t0 + datetime.timedelta(milliseconds=fin):
            if typ == 'Robot':
                indexout.append(t)
                RobotSpeak.append(1)
                Robotdur.append(float(fin - deb + 2) / 1000)
                Robotdurt.append((t - datetime.timedelta(milliseconds=deb + 1) - t0).total_seconds())
                userdur.append(0)
                userdurt.append(0)
            elif typ == 'User':
                indexout.append(t)
                RobotSpeak.append(0)
                Robotdur.append(0)
                Robotdurt.append(0)
                userdur.append(float(fin - deb + 2) / 1000)
                userdurt.append((t - datetime.timedelta(milliseconds=deb + 1) - t0).total_seconds())
            else:
                indexout.append(t)
                RobotSpeak.append(np.nan)
                Robotdur.append(np.nan)
                userdur.append(np.nan)
                Robotdurt.append(np.nan)
                userdurt.append(np.nan)
        else:
            indexout.append(t)
            RobotSpeak.append(np.nan)
            Robotdur.append(np.nan)
            userdur.append(np.nan)
            Robotdurt.append(np.nan)
            userdurt.append(np.nan)
        t = t + pd.to_timedelta(stepSize, unit='ms')
    dfout = pd.concat([dfout, pd.DataFrame(data=RobotSpeak, columns=['IsListeningToRobot'], index=indexout)], axis=1)
    dfout = pd.concat([dfout, pd.DataFrame(data=Robotdur, columns=['RobotSpeakDur'], index=indexout)], axis=1)
    dfout = pd.concat([dfout, pd.DataFrame(data=userdur, columns=['RobotListenDur'], index=indexout)], axis=1)
    return dfout

def annotateddata(deb, fin, typinteraction, blockSize):
    """
        Make engagement segment as binary:
            BD = 1
            Others (NBD,SED,EBD,...) = 0
    """
    Disengage = {1: 'SED', 2: 'EBD'}
    Disengage_BD = {19: 'TBD', 20: 'BD'}
    dfout = pd.DataFrame()
    indexout = []
    eng = []
    t = deb
    while t <= fin - pd.to_timedelta(blockSize, unit='ms'):
        indexout.append(t)
        if typinteraction in Disengage.values():
            eng.append(dict(zip(Disengage.values(), Disengage.keys()))[typinteraction])
        elif typinteraction in Disengage_BD.values():
            eng.append(dict(zip(Disengage_BD.values(), Disengage_BD.keys()))[typinteraction])
        elif typinteraction == 'NoInteraction':
            eng.append(-1)
        elif typinteraction == 'Mono':
            eng.append(10)
        elif typinteraction == 'Multi':
            eng.append(11)
        else:
            eng.append(0)
        t = t + pd.to_timedelta(blockSize, unit='ms')
    dfout = pd.concat([dfout, pd.DataFrame(data=eng, columns=['eng'], index=indexout)], axis=1)    
    return dfout, t

def get_annotation_all(eaf, blockSize, dfdata, ignore_length_sec = 0, useALL=True):
    """
    Convert ELAN file .eaf to dataframe
    """
    t0 = dfdata.index[0]
    tN = dfdata.index[-1]
    nbinteraction = 0
    dfeng = pd.DataFrame()
    deb2 = t0
    for i in range(len(sorted(eaf.get_annotation_data_for_tier('Interaction')))):
        nbinteraction+=1
        annotation=False
        deb = t0+datetime.timedelta(milliseconds=sorted(eaf.get_annotation_data_for_tier('Interaction'))[i][0])
        dfout, deb = annotateddata(deb2, deb, 'NoInteraction', blockSize)
        dfeng = pd.concat([dfeng, dfout])
        fin = t0+datetime.timedelta(milliseconds=sorted(eaf.get_annotation_data_for_tier('Interaction'))[i][1])
        typinteraction = sorted(eaf.get_annotation_data_for_tier('Interaction'))[i][2]
        if typinteraction == 'Mono' or useALL:
            if len(sorted(eaf.get_annotation_data_between_times('Engagement',
                                                                sorted(eaf.get_annotation_data_for_tier('Interaction'))[i][0],
                                                                sorted(eaf.get_annotation_data_for_tier('Interaction'))[i][1]))) > 0:
                print('Engagement decrease')
                for j in  range(len(sorted(eaf.get_annotation_data_between_times('Engagement',
                                                                                 sorted(eaf.get_annotation_data_for_tier('Interaction'))[i][0],
                                                                                 sorted(eaf.get_annotation_data_for_tier('Interaction'))[i][1])))):

                    sed=sorted(eaf.get_annotation_data_between_times('Engagement',
                                                                     sorted(eaf.get_annotation_data_for_tier('Interaction'))[i][0],
                                                                     sorted(eaf.get_annotation_data_for_tier('Interaction'))[i][1]))[j]
                    debBD = t0 + datetime.timedelta(milliseconds=sed[0])
                    finBD = t0 + datetime.timedelta(milliseconds=sed[1])
                    typBD = sed[2]
                    if j+1 < len(sorted(eaf.get_annotation_data_between_times('Engagement', 
                                                                              sorted(eaf.get_annotation_data_for_tier('Interaction'))[i][0],
                                                                              sorted(eaf.get_annotation_data_for_tier('Interaction'))[i][1]))):
                        debBDnext = t0 + datetime.timedelta(milliseconds=sorted(eaf.get_annotation_data_between_times('Engagement', 
                                                                                                                      sorted(eaf.get_annotation_data_for_tier('Interaction'))[i][0],
                                                                                                                      sorted(eaf.get_annotation_data_for_tier('Interaction'))[i][1]))[j+1][0])
                        typBDnext = sorted(eaf.get_annotation_data_between_times('Engagement',
                                                                                 sorted(eaf.get_annotation_data_for_tier('Interaction'))[i][0],
                                                                                 sorted(eaf.get_annotation_data_for_tier('Interaction'))[i][1]))[j+1][2]
                        if debBDnext-finBD < pd.Timedelta(ignore_length_sec,'s') and typBDnext not in ['TBD','BD']:
                            j=j+1
                            print('old:',debBD,finBD,typBD,'\tignoring:',debBDnext)
                            finBD = t0 + datetime.timedelta(milliseconds=sorted(
                                eaf.get_annotation_data_between_times('Engagement',
                                                                      sorted(eaf.get_annotation_data_for_tier('Interaction'))[i][0],
                                                                      sorted(eaf.get_annotation_data_for_tier('Interaction'))[i][1]))[j][1])
                            typBD = sorted(eaf.get_annotation_data_between_times('Engagement', sorted(eaf.get_annotation_data_for_tier('Interaction'))[i][0],sorted(eaf.get_annotation_data_for_tier('Interaction'))[i][1]))[j][2]
                            print('Ignored engaged segment between 2 SED where its length is less than 1s')
                            print('New:',debBD,finBD,typBD)
                        elif debBDnext-finBD < pd.Timedelta(ignore_length_sec,'s') and typBDnext in ['TBD','BD']:
                            print('old:',debBD,finBD,typBD,'\tignoring:',debBDnext)
                            finBD = t0 + datetime.timedelta(milliseconds=sorted(
                                eaf.get_annotation_data_between_times('Engagement',
                                                                      sorted(eaf.get_annotation_data_for_tier('Interaction'))[i][0],
                                                                      sorted(eaf.get_annotation_data_for_tier('Interaction'))[i][1]))[j+1][0])
                            typBD = sorted(eaf.get_annotation_data_between_times('Engagement', 
                                                                                 sorted(eaf.get_annotation_data_for_tier('Interaction'))[i][0],
                                                                                 sorted(eaf.get_annotation_data_for_tier('Interaction'))[i][1]))[j][2]
                            print('Ignored engaged segment between SED and BD where its length is less than 1s')
                            print('New:',debBD,finBD,typBD)
                            
                    if deb <= debBD and debBD < fin:# and typBD!='BD':
                        annotation=True
                        dfout, finEng = annotateddata(deb, debBD, 'Eng', blockSize)
                        dfeng = pd.concat([dfeng, dfout])
                        dfout, deb = annotateddata(finEng, finBD, typBD, blockSize)
                        dfeng = pd.concat([dfeng, dfout])
                dfout, deb2 = annotateddata(deb, fin, 'Eng', blockSize)
                dfeng = pd.concat([dfeng, dfout])
            else:
                annotation=True
                print('No-Breakdown: SED')
                dfout, deb2 = annotateddata(deb, fin, 'Eng', blockSize)
                dfeng = pd.concat([dfeng, dfout])
                        
            if annotation==False:
                print('No-Breakdown')
                dfout, deb2 = annotateddata(deb, fin, 'Eng', blockSize)
                dfeng = pd.concat([dfeng, dfout])
        else:
            print(user,'ignored annotation. It is a multiparty interaction!')
    dfout, deb2 = annotateddata(deb2, tN, 'NoInteraction', blockSize)
    dfeng = pd.concat([dfeng, dfout])
    return dfeng, nbinteraction, t0, tN

def xy_engtestdata(dfdata, topicsnames, eaf, eaf2, stepSize, blockSize, blockSizeEng, listengBD, firstSecs, lastSecs,
                   hist=False):
    """
        Compute input and output data
    """
    bd = 20
    tbd = 19
    eng = 0
    blockSizeHist = listengBD[-1]
    start_time0 = time.time()
    dfeng_disengage, nbinteract, t01, tN1 = get_annotation_all(eaf, blockSize, dfdata)
    dfeng_disengage2, nbinteract2, t02, tN2 = get_annotation_all(eaf2, blockSize, dfdata)
    tsyn = min(dfeng_disengage.index[0],dfeng_disengage2.index[0]) + pd.to_timedelta(firstSecs, unit='s')
    indexout = []
    vectors = []
    dfInteract = pd.DataFrame()
    if True:
        dfdata_interact1 = dfeng_disengage.copy()
        dfdata_interact2 = dfeng_disengage2.copy()
        mask = ((dfeng_disengage.index >= max(dfeng_disengage.index[0],dfeng_disengage2.index[0])) & 
                (dfeng_disengage.index <= min(dfeng_disengage.index[-1],dfeng_disengage2.index[-1])))
        dfdata_interact1 = dfeng_disengage.loc[mask]
        dfdata_interact1.columns=['anno1']
        mask2 = ((dfeng_disengage2.index >= max(dfeng_disengage.index[0],dfeng_disengage2.index[0])) & 
                 (dfeng_disengage2.index <= min(dfeng_disengage.index[-1],dfeng_disengage2.index[-1])))
        dfdata_interact2 = dfeng_disengage2.loc[mask2]
        dfdata_interact2.columns=['anno2']
        first = firstSecs
        if (len(dfdata_interact1[dfdata_interact1['anno1'] >= bd]) > 0 and 
            len(dfdata_interact2[dfdata_interact2['anno2'] >= bd]) > 0):
            tN = min(dfdata_interact1[dfdata_interact1['anno1'] >= bd].index[0], 
                     dfdata_interact2[dfdata_interact2['anno2'] >= bd].index[0])
        elif (len(dfdata_interact1[dfdata_interact1['anno1'] >= bd]) > 0):
            tN = dfdata_interact1[dfdata_interact1['anno1'] >= bd].index[0]
        elif (len(dfdata_interact2[dfdata_interact2['anno2'] >= bd]) > 0):
            tN = dfdata_interact2[dfdata_interact2['anno2'] >= bd].index[0]
        else:
            tN = min(dfdata_interact1.index[-1],dfdata_interact2.index[-1])
        if max(dfdata_interact1[dfdata_interact1['anno1'] >= eng].index[0],dfdata_interact2[dfdata_interact2['anno2'] >= eng].index[0]) + pd.to_timedelta(first, unit='s') >= tN:
            t = max(dfdata_interact1[dfdata_interact1['anno1'] >= eng].index[0],dfdata_interact2[dfdata_interact2['anno2'] >= eng].index[0])
        else:
            if (len(dfdata_interact1[dfdata_interact1['anno1'] >= bd].dropna()) > 0 and 
                len(dfdata_interact2[dfdata_interact2['anno2'] >= bd].dropna()) > 0):
                t = min(max(dfdata_interact1[dfdata_interact1['anno1'] >= eng].index[0],dfdata_interact2[dfdata_interact2['anno2'] >= eng].index[0]) + pd.to_timedelta(first, unit='s'),
                        max(max(dfdata_interact1[dfdata_interact1['anno1'] >= eng].index[0],dfdata_interact2[dfdata_interact2['anno2'] >= eng].index[0]),
                            min(dfdata_interact1[dfdata_interact1['anno1'] >= bd].dropna().index[0],
                                dfdata_interact2[dfdata_interact2['anno2'] >= bd].dropna().index[0]) - pd.to_timedelta(blockSizeHist, unit='ms')
                            ))
            elif (len(dfdata_interact1[dfdata_interact1['anno1'] >= bd]) > 0):
                t = min(max(dfdata_interact1[dfdata_interact1['anno1'] >= eng].index[0],dfdata_interact2[dfdata_interact2['anno2'] >= eng].index[0]) + pd.to_timedelta(first, unit='s'),
                        max(max(dfdata_interact1[dfdata_interact1['anno1'] >= eng].index[0],dfdata_interact2[dfdata_interact2['anno2'] >= eng].index[0]),
                            dfdata_interact1[dfdata_interact1['anno1'] >= bd].dropna().index[0] - pd.to_timedelta(blockSizeHist, unit='ms')
                            ))
            elif (len(dfdata_interact2[dfdata_interact2['anno2'] >= bd]) > 0):
                t = min(max(dfdata_interact1[dfdata_interact1['anno1'] >= eng].index[0],dfdata_interact2[dfdata_interact2['anno2'] >= eng].index[0]) + pd.to_timedelta(first, unit='s'),
                        max(max(dfdata_interact1[dfdata_interact1['anno1'] >= eng].index[0],dfdata_interact2[dfdata_interact2['anno2'] >= eng].index[0]),
                            dfdata_interact2[dfdata_interact2['anno2'] >= bd].dropna().index[0] - pd.to_timedelta(blockSizeHist, unit='ms')
                            ))
            else:
                t = max(dfdata_interact1[dfdata_interact1['anno1'] >= eng].index[0],dfdata_interact2[dfdata_interact2['anno2'] >= eng].index[0]) + pd.to_timedelta(first, unit='s')
        if lastSecs > 0:
            t = max(t, tN - pd.to_timedelta(lastSecs, unit='s'))
        deb = t
        if (len(dfdata_interact1[dfdata_interact1['anno1'] >= bd]) > 0 and 
            len(dfdata_interact2[dfdata_interact2['anno2'] >= bd]) > 0):
            bd0 = min(dfdata_interact1[dfdata_interact1['anno1'] == bd].index[0], 
                     dfdata_interact2[dfdata_interact2['anno2'] == bd].index[0])
        elif (len(dfdata_interact1[dfdata_interact1['anno1'] >= bd]) > 0):
            bd0 = dfdata_interact1[dfdata_interact1['anno1'] == bd].index[0]
        elif (len(dfdata_interact2[dfdata_interact2['anno2'] >= bd]) > 0):
            bd0 = dfdata_interact2[dfdata_interact2['anno2'] == bd].index[0]
        else:
            bd0 = min(dfdata_interact1.index[-1],dfdata_interact2.index[-1])
        if (len(dfdata_interact1[dfdata_interact1['anno1'] == tbd]) > 0 and 
            len(dfdata_interact2[dfdata_interact2['anno2'] == tbd]) > 0):
            tbd0 = min(dfdata_interact1[dfdata_interact1['anno1'] == tbd].index[0], 
                     dfdata_interact2[dfdata_interact2['anno2'] == tbd].index[0])
            tbdN = max(dfdata_interact1[dfdata_interact1['anno1'] == tbd].index[-1], 
                     dfdata_interact2[dfdata_interact2['anno2'] == tbd].index[-1])
            locInteract = (((dfdata_interact1.index < tbd0) | (dfdata_interact1.index > tbdN)) & (dfdata_interact1.index < bd0) & 
                           ((dfdata_interact2.index < tbd0) | (dfdata_interact2.index > tbdN)) & (dfdata_interact2.index < bd0))
            locdataInteract = ((dfdata.index < tbd0) | (dfdata.index > tbdN)) & (dfdata.index >= deb) & (
                dfdata.index < bd0)
        elif (len(dfdata_interact1[dfdata_interact1['anno1'] == tbd]) > 0):
            tbd0 = dfdata_interact1[dfdata_interact1['anno1'] == tbd].index[0]
            tbdN = dfdata_interact1[dfdata_interact1['anno1'] == tbd].index[-1]
            locInteract = ((dfdata_interact1.index < tbd0) | (dfdata_interact1.index > tbdN)) & (
                dfdata_interact1.index < bd0)
            locdataInteract = ((dfdata.index < tbd0) | (dfdata.index > tbdN)) & (dfdata.index >= deb) & (
                dfdata.index < bd0)
        elif (len(dfdata_interact2[dfdata_interact2['anno2'] == tbd]) > 0):
            tbd0 = dfdata_interact2[dfdata_interact2['anno2'] == tbd].index[0]
            tbdN = dfdata_interact2[dfdata_interact2['anno2'] == tbd].index[-1]
            locInteract = ((dfdata_interact2.index < tbd0) | (dfdata_interact2.index > tbdN)) & (
                dfdata_interact2.index < bd0)
            locdataInteract = ((dfdata.index < tbd0) | (dfdata.index > tbdN)) & (dfdata.index >= deb) & (
                dfdata.index < bd0)
        else:
            locInteract = (((dfdata_interact1.index >= deb) & (dfdata_interact1.index < bd0)) &
                           ((dfdata_interact2.index >= deb) & (dfdata_interact2.index < bd0)))
            locdataInteract = (dfdata.index >= deb) & (dfdata.index < bd0)
        df_tmp = pd.concat([dfdata_interact1.loc[locInteract], dfdata_interact2.loc[locInteract]],axis=1)
        dfdatasubinteract = pd.merge(dfdata.loc[locdataInteract], df_tmp, left_index=True,
                                     right_index=True, how='outer')
        dfInteract = pd.concat([dfInteract, dfdatasubinteract])
        while t <= tN:
            if not hist:
                if ~np.isnan(dfdatasubinteract.loc[t - pd.to_timedelta(blockSizeHist, unit='ms'):t].ix[:,
                             dfdatasubinteract.columns != 'eng'].values).all():
                    indexout.append(tsyn)
                    xhist = dfdatasubinteract.loc[t - pd.to_timedelta(blockSizeHist, unit='ms'):t].ix[:,
                            dfdatasubinteract.columns != 'eng'].mean().values
                    vectors.append(np.concatenate((xhist, np.array([dfeng_disengage.loc[
                                                                    t - pd.to_timedelta(blockSizeEng, unit='ms'):t
                                                                    ].values[0][0]])),
                                                  axis=0))
            else:
                if ~np.all(dfdatasubinteract.loc[t - pd.to_timedelta(blockSizeHist, unit='ms'):t].isna()):
                    xhist = dfdatasubinteract.loc[t - pd.to_timedelta(blockSizeHist, unit='ms'):t][[c 
                                                  for c in dfdatasubinteract.columns if c not in ['anno1','anno2']]].values
                    if xhist.shape[0] == (listengBD[-1] / blockSize) + 1:
                        lbl = dfdatasubinteract.loc[t - pd.to_timedelta(blockSizeEng, unit='ms'):t][['anno1','anno2']].values[0]#.max().max()
                        indexout.append(tsyn)
                        vectors.append(np.concatenate(
                                (xhist.reshape(xhist.shape[0] * xhist.shape[1]), 
                                 lbl
                                 ),axis=0))

            t = t + pd.to_timedelta(stepSize, unit='ms')
            tsyn = tsyn + pd.to_timedelta(stepSize, unit='ms')
            
    if hist:
        columns_names = []
        for c in range((listengBD[-1]) // blockSize, -1, -1):
            if c != 0:
                columns_names += dfdatasubinteract[[col for col in dfdatasubinteract.columns 
                                                    if col not in ['anno1','anno2']]].add_suffix(
                                                    '_t' + '-' + str(c)).columns.tolist()
            else:
                columns_names += dfdatasubinteract[[col for col in dfdatasubinteract.columns 
                                                    if col not in ['anno1','anno2']]].add_suffix(
                                                    '_t').columns.tolist()
        columns_names += dfdatasubinteract[['anno1','anno2']].columns.tolist()
    else:
        columns_names = dfdatasubinteract.columns.values
    print("Done in %0.3fs" % (time.time() - start_time0))
    dfdatahist = pd.DataFrame(data=vectors, columns=columns_names, index=indexout)
    return dfInteract, dfdatahist, nbinteract, nbinteract2

def load_data(tau=5, eta=2):
    """
    Returns:
    - The list of users data, X_all_users, where each entry of the list is an array
      of shape (nb_seq x seq_length x nb_feat),
    - The list of labels, Y_all_users, where each entry of the list is an array of labels
      of size nb_seq.
    """
    # 1. Load user data
    # The result is a list of user data (each entry of the list contains one user's data)
    # Each entry is a list (pandas array) of sequences of the same length
    # Each sequence is represented as a vector of size (seq_length x nb_feat)
    lastSecs = 0
    # For tau & eta, we convert to ms then divide by 2. Not quite intuitive but this may have to do
    # with the fact 
    deltaT = tau * 1000 # tau
    blockSizeEng = eta * 1000 # eta
    
    # 5000 (tau = 10) : 943 rows
    # 2500 (tau = 5) : 958 rows
    # 1000 (tau = 2) : 965 rows
    # 0 (tau = 0) : 967 rows
    
    direaf = './Volumes/LaCie2/Annotation/eaf_turnTalk2_GoogleASR/'
    direaf2 = './Volumes/LaCie2/Annotation/eaf_IMBY_turnTalk2_GoogleASR/'
    dirinput = './Volumes/LaCie1/hdf_TemporalIntegration_500block_500step_add/'
    dirwork = './models/'
    idx = '0,1,2,3,4,5,6,7,11'
    dirpredeaf = direaf

    try:
        os.stat(dirwork)
    except:
        os.mkdir(dirwork)

    start_time = time.time()

    firstSecs = 0 # 60
    listengBD = [deltaT]
    TurnTalk = True
    # Temporal integration parameters
    stepSize = 500
    blockSize = 500

    topics = [
        '/sonar/front/data',          # 0
        '/face/Distance',             # 1
        '/face/PositionInTorsoFrame', # 2
        '/face/GazeDirection',        # 3
        '/face/IsLookingAtRobot',     # 4
        '/face/HeadAngles',           # 5
        '/face/SmileProperties',      # 6
        '/face/ExpressionProperties', # 7
        '/face/FacialPartsProperties',# 8
        '/face/EngagementZone',       # 9
        '/laser/data',                # 10
        '/audio/speechfeatures',      # 11
        '/face/GenderProperties',     # 12
        '/OpenFace'                   # 13
    ]

    topicsnames = []
    for i in idx.split(','):
        topicsnames = topicsnames + [topics[int(i)]]

    topicfeat = str(topicsnames).replace('[', '').replace(']', '').replace('\'', '').split('/')

    nmtpc = ''
    for tpc in range(1, len(topicfeat)):
        nmtpc = nmtpc + topicfeat[tpc][0]

    print(nmtpc, ":", topicsnames)
    
    start_time_0 = time.time()
    nbuser = 0
    nbuserInteract = 0
    missuser = []
    dfInteract = pd.DataFrame()
    dfInteract_all = {}
    
    X_all_users = []
    Y_all_users = []
    
    for fileeaf in sorted(glob.glob(direaf2 + "/*.eaf")):
        user = os.path.splitext(os.path.basename(fileeaf))[0]
        print(user)
        fileh5 = dirinput + user + '.h5'

        try:
            store = pd.HDFStore(fileh5)
            if os.path.exists(direaf2 + user + '.eaf'):
                eaf2 = pympi.Elan.Eaf(direaf2 + user +'.eaf')
            else:
                eaf2 = pympi.Elan.Eaf(direaf + user +'.eaf')
            if os.path.exists(direaf + user + '.eaf'):
                eaf = pympi.Elan.Eaf(direaf + user + '.eaf')
            else:
                eaf = pympi.Elan.Eaf(direaf2 + user + '.eaf')
            dfdata = pd.DataFrame()
            for stream in topicsnames:
                use_col = [column for column in store[stream].columns.values if 'de' not in column]
                dfuser = store[stream][use_col]
                if '/face/HeadAngles' == stream:
                    dfuser.columns = 'Head_' + dfuser.columns
                elif '/face/GazeDirection' == stream:
                    dfuser.columns = 'Gaze_' + dfuser.columns
                elif '/laser/data' == stream:
                    ignore_laser_noise = [idlsr for idlsr in store[stream].columns if
                                          (int(idlsr[0:2].replace('_', '')) in range(23, 38))]
                    dfuser = store[stream][ignore_laser_noise]
                elif '/OpenFace' == stream:
                    used_feat = [c for c in store[stream].columns if
                                 'gaze_angle' in c or 'pose_R' in c or 'AU' in c]
                    dfuser = store[stream][used_feat]
                dfdata = pd.concat([dfdata, dfuser], axis=1)

            if TurnTalk:
                df = robot_is_speaking(eaf, dfdata, blockSize)
                dfdata = pd.concat([dfdata, df], axis=1)

            dfinteract_hist, dfdatahist, nbinteract, nbinteract2 = xy_engtestdata(dfdata, topicsnames, eaf, eaf2, 
                                                                                  stepSize, blockSize, blockSizeEng, 
                                                                                  listengBD, firstSecs, lastSecs, True)

            dfdatahist = dfdatahist.drop(dfdatahist[dfdatahist['anno1']==-1].index)
            dfdatahist = dfdatahist.drop(dfdatahist[dfdatahist['anno2']==-1].index)
            dfdatahist.loc[dfdatahist[dfdatahist['anno1'] >=2].index,'anno1'] = 1
            dfdatahist.loc[dfdatahist[dfdatahist['anno2'] >=2].index,'anno2'] = 1

            dfInteract = pd.concat([dfInteract, dfdatahist])
            dfInteract_all[nbuser] = dfdatahist

            nbuser += 1
            nbuserInteract += nbinteract
            print(str(nbuser) +" interctions => "+str(float(nbuser) / len(glob.glob(direaf2 + "/*.eaf")) * 100) + " %")
        except:
            print("ERROR: topics not exist in " + store.filename)
            missuser.append(user)
        store.close()

    print("NbUsers", nbuser, "NbInteractions", nbuserInteract)
    print("Done in %0.3fs" % (time.time() - start_time_0))
    
    # 2. Reshape users data such that for each user, the corresponding data
    # has the shape (nb_seq x seq_length x nb_feat)
    for user_id in range(len(dfInteract_all)):
        userdata = dfInteract_all[list(dfInteract_all.keys())[user_id]]
        userdata = userdata[userdata['anno1'] == userdata["anno2"]]
        userdata = userdata.dropna(subset=['anno1', 'anno2'])
        
        # All data sequences corresponding to user_id
        X = userdata.drop(['anno1', 'anno2'], axis=1).values.reshape(
            userdata.shape[0],
            (listengBD[-1] + blockSize) // blockSize, 
            userdata.drop(['anno1', 'anno2'], axis=1).values.shape[1] // ((listengBD[-1] + blockSize) // blockSize))
        
        # Labels of all sequences corresponding to user_id (same length as X)
        Y = userdata['anno1'].values
        
        # If data frame not empty, add it to the data set
        if not userdata.empty:
            X_all_users.append(X)
            Y_all_users.append(Y)

    return np.asarray(X_all_users), np.asarray(Y_all_users)

# Preprocess data then save the result as .npy files
tau = 20
eta = 2
X_all_users, Y_all_users = load_data(tau, eta)
np.save("X_all_users_tau_" + str(tau), X_all_users)
np.save("Y_all_users_tau_" + str(tau) + "_eta_" + str(eta), Y_all_users)