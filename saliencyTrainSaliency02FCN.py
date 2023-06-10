from definitions import *
from configTrainSaliency02FCN import *
from nets import CNNmodelKeras, FCNmodel

# saliency_model = CNNmodelKeras(img_size, num_channels, num_classes, type)
saliency_model = FCNmodel(img_size, 64, num_channels, num_classes, type)

saliency_model.compile(loss=keras.losses.MeanSquaredError(), optimizer='adam',
                       metrics=[keras.metrics.RootMeanSquaredError()])

train_data = []
train_labels = []
train_labels_patch = []
trainSet = ['armchair']
for modelName in trainSet:
    # ======Model information=====================================================================
    mModelSrc = rootdir + modelsDir + modelName + '.obj'
    print(modelName)
    if mode == "MESH":
        mModel = loadObj(mModelSrc)
        updateGeometryAttibutes(mModel, useGuided=useGuided, numOfFacesForGuided=patchSizeGuided, computeDeltas=False,
                                computeAdjacency=False, computeVertexNormals=False)
    if mode == "PC":
        mModel = loadObjPC(mModelSrc, nn=pointcloudnn, simplify=presimplification)
        V, inds = computePointCloudNormals(mModel, pointcloudnn)
        exportPLYPC(mModel, modelsDir + modelName + '_pcnorm_conf.ply')

    gtdata = np.genfromtxt(rootdir + modelsDir + modelName + groundTruthKeyword + '.csv', delimiter=',')

    # #saliencyValue=saliencyValue/np.max(saliencyValue)
    print('Saliency ground truth data')
    if type == 'continuous':
        saliencyValues = gtdata.tolist()
    if type == 'discrete':
        saliencyValues = []
        for s in gtdata.tolist():
            v = int((num_classes - 1) * s)
            saliencyValues.append(v)

    if mode == "MESH":
        iLen = len(mModel.faces)
        patches = [neighboursByFace(mModel, i, numOfElements)[0] for i in range(0, len(mModel.faces))]
        # Rotation and train data formulation===============================================================================
        for i, p in enumerate(patches):
            print(i)
            patchFacesOriginal = [mModel.faces[i] for i in p]
            positionsPatchFacesOriginal = np.asarray([pF.centroid for pF in patchFacesOriginal])
            normalsPatchFacesOriginal = np.asarray([pF.faceNormal for pF in patchFacesOriginal])
            # vec = np.mean(np.asarray(
            #         [fnm.faceNormal for fnm in [mModel.faces[j] for j in neighboursByFace(mModel, i, 4)[0]]]
            #     ), axis=0)
            vec = np.mean(np.asarray([fnm.area * fnm.faceNormal for fnm in patchFacesOriginal]), axis=0)
            # vec = mModel.faces[i].faceNormal
            vec = vec / np.linalg.norm(vec)
            axis, theta = computeRotation(vec, target)
            normalsPatchFacesOriginal = rotatePatch(normalsPatchFacesOriginal, axis, theta)
            normalsPatchFacesOriginalR = normalsPatchFacesOriginal.reshape((patchSide, patchSide, 3))
            if reshapeFunction == "hilbert":
                for hci in range(np.shape(I2HC)[0]):
                    normalsPatchFacesOriginalR[I2HC[hci, 0], I2HC[hci, 1], :] = normalsPatchFacesOriginal[:,
                                                                                HC2I[I2HC[hci, 0], I2HC[hci, 1]]]
            train_data.append((normalsPatchFacesOriginalR + 1.0 * np.ones(np.shape(normalsPatchFacesOriginalR))) / 2.0)

    if mode == "PC":
        iLen = len(mModel.vertices)
        patches = [neighboursByVertex(mModel, i, numOfElements)[0] for i in range(0, len(mModel.vertices))]
        # patches = np.random.choice(patches, numOfElements, replace=False)
        for i, p in enumerate(patches):
            print(i)
            patchVerticesOriginal = [mModel.vertices[i] for i in p]
            normalsPatchVerticesOriginal = np.asarray([pF.normal for pF in patchVerticesOriginal])
            vec = np.mean(np.asarray([fnm.normal for fnm in patchVerticesOriginal]), axis=0)
            vec = vec / np.linalg.norm(vec)
            axis, theta = computeRotation(vec, target)
            normalsPatchVerticesOriginal = rotatePatch(normalsPatchVerticesOriginal, axis, theta)
            normalsPatchVerticesOriginalR = normalsPatchVerticesOriginal.reshape((patchSide, patchSide, 3))
            if reshapeFunction == "hilbert":
                for hci in range(np.shape(I2HC)[0]):
                    normalsPatchVerticesOriginalR[I2HC[hci, 0], I2HC[hci, 1], :] = normalsPatchVerticesOriginal[:,
                                                                                   HC2I[I2HC[hci, 0], I2HC[hci, 1]]]
            train_data.append(
                (normalsPatchVerticesOriginalR + 1.0 * np.ones(np.shape(normalsPatchVerticesOriginalR))) / 2.0)

    for i, p in enumerate(patches):
        saliencyValuesPerPatch = np.asarray([saliencyValues[i] for i in p])
        saliencyValuesPerPatchR = saliencyValuesPerPatch.reshape((patchSide, patchSide, 1))
        if reshapeFunction == "hilbert":
            for hci in range(np.shape(I2HC)[0]):
                saliencyValuesPerPatchR[I2HC[hci, 0], I2HC[hci, 1], 0] = saliencyValuesPerPatch[
                    HC2I[I2HC[hci, 0], I2HC[hci, 1]]]
        train_labels_patch.append(saliencyValuesPerPatchR)

# Dataset and labels summarization ========================================================================
if type == 'continuous':
    seppoint = int(0.9 * np.shape(train_data)[0])

    train_data = np.asarray(train_data)
    train_labels = np.asarray(train_labels_patch)

    X = train_data[:seppoint]
    X_test = train_data[seppoint:]
    Y = np.asarray(train_labels[:seppoint])
    Y_test = np.asarray(train_labels[seppoint:])
    data_train = X
    data_test = X_test
    label_train = Y
    label_test = Y_test



saliency_model.summary()
saliency_model_train = saliency_model.fit(x=data_train, y=label_train, batch_size=batch_size, epochs=numEpochs,
                                          verbose=1)
saliency_model.save(rootdir+sessionsDir + keyTrain + '.h5')
