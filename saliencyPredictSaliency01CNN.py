from definitions import *
from configPredictSaliency01CNN import *
from nets import CNNmodelKeras
import convert
import argparse
import open3d as o3d
import yaml
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str)
args = parser.parse_args()
modelName = 'casting'


saliency_model = CNNmodelKeras(img_size, num_channels, num_classes, type)
saliency_model.load_weights(rootdir+sessionsDir + keyTrain + ".h5")
# from keras import backend as K
# print(K.tensorflow_backend._get_available_gpus())
print('Predict')
# ======Model information==============================================================================



mModelSrc = rootdir + modelsDir + modelName + '.obj'
print(modelName)
if mode == "MESH":
    mModel = loadObj(mModelSrc)
    updateGeometryAttibutes(mModel, useGuided=useGuided, numOfFacesForGuided=patchSizeGuided, computeDeltas=False,
                            computeAdjacency=False, computeVertexNormals=False)
    iLen = len(mModel.faces)




if mode == "PC":
    mModel = loadObjPC(mModelSrc, nn=pointcloudnn, simplify=presimplification)
    V, inds = computePointCloudNormals(mModel, pointcloudnn)
    iLen = len(mModel.vertices)
predict_data = np.empty([iLen, patchSide, patchSide, 3])
prediction = np.empty([iLen, 1])
print('Start')
print("Loading model:" + keyTrain)
# subset = random.sample(list(range(iLen)), int(0.1 * iLen))
# subset.sort()
# for ipointer in subset:
for ipointer in range(0, iLen, batchsize):
    pall = []
    istart = ipointer
    iend = min(ipointer + batchsize, iLen - 1)
    for i in range(istart, iend):
        if i % 2000 == 0:
            print('Extract patch information : ' + str(
                np.round((100 * i / iLen), decimals=2)) + ' ' + '%')

        if mode == "MESH":
            patchFacesOriginal = [mModel.faces[i] for i in neighboursByFace(mModel, i, numOfElements)[0]]
            # patchFacesOriginal = [mModel.faces[i] for i in mModel.faces[i].neighbouringFaceIndices64[:numOfElements]]
            normalsPatchFacesOriginal = np.asarray([pF.faceNormal for pF in patchFacesOriginal])
            if doRotate:
                vec = np.mean(np.asarray(
                    [fnm.faceNormal for fnm in [mModel.faces[j] for j in neighboursByFace(mModel, i, 4)[0]]]),
                    axis=0)
                # vec = np.mean(np.asarray([fnm.area * fnm.faceNormal for fnm in patchFacesOriginal]), axis=0)
                # vec = mModel.faces[i].faceNormal
                vec = vec / np.linalg.norm(vec)
                target = np.array([0.0, 1.0, 0.0])
                axis, theta = computeRotation(vec, target)
                normalsPatchFacesOriginal = rotatePatch(normalsPatchFacesOriginal, axis, theta)
            normalsPatchFacesOriginalR = normalsPatchFacesOriginal.reshape((patchSide, patchSide, 3))
            if reshapeFunction == "hilbert":
                for hci in range(np.shape(I2HC)[0]):
                    hicoord = HC2I[I2HC[hci, 0], I2HC[hci, 1]]
                    hx = I2HC[hci, 0]
                    hy = I2HC[hci, 1]
                    normalsPatchFacesOriginalR[hx, hy, :] = normalsPatchFacesOriginal[:, hicoord]
                    # normalsPatchFacesOriginalR[hc[0], hc[1], :] = normalsPatchFacesOriginal[:, hCoords.index(hc)]
            pIn = (normalsPatchFacesOriginalR + 1.0 * np.ones(np.shape(normalsPatchFacesOriginalR))) / 2.0
            pall.append(pIn)
        if mode == "PC":
            patchVerticesOriginal = [mModel.vertices[i] for i in neighboursByVertex(mModel, i, numOfElements)[0]]
            # patchFacesOriginal = [mModel.faces[i] for i in mModel.faces[i].neighbouringFaceIndices64[:numOfElements]]
            normalsPatchVerticesOriginal = np.asarray([pV.normal for pV in patchVerticesOriginal])
            for k in range(0, 2):
                normalsPatchVerticesOriginal[k, :] = np.array([0.0, 1.0, 0.0])
            if doRotate:
                vec = np.mean(np.asarray(
                    [vnm.normal for vnm in [mModel.vertices[j] for j in neighboursByVertex(mModel, i, 4)[0]]]),
                    axis=0)
                # vec = np.mean(np.asarray([fnm.area * fnm.faceNormal for fnm in patchFacesOriginal]), axis=0)
                # vec = mModel.faces[i].faceNormal
                vec = vec / np.linalg.norm(vec)
                target = np.array([0.0, 1.0, 0.0])
                axis, theta = computeRotation(vec, target)
                normalsPatchVerticesOriginal = rotatePatch(normalsPatchVerticesOriginal, axis, theta)
            normalsPatchVerticesOriginalR = normalsPatchVerticesOriginal.reshape((patchSide, patchSide, 3))
            if reshapeFunction == "hilbert":
                for hci in range(np.shape(I2HC)[0]):
                    hicoord = HC2I[I2HC[hci, 0], I2HC[hci, 1]]
                    hx = I2HC[hci, 0]
                    hy = I2HC[hci, 1]
                    normalsPatchVerticesOriginalR[hx, hy, :] = normalsPatchVerticesOriginal[:, hicoord]
                    # normalsPatchFacesOriginalR[hc[0], hc[1], :] = normalsPatchFacesOriginal[:, hCoords.index(hc)]
            pIn = (normalsPatchVerticesOriginalR + 1.0 * np.ones(np.shape(normalsPatchVerticesOriginalR))) / 2.0
            pall.append(pIn)
    x_batch = np.asarray(pall)
    result = saliency_model.predict(x_batch)
    # a = result[0].tolist()
    # r = 0
    # # Finding the maximum of all outputs
    # max1 = max(a)
    # prediction[i]=index1
    # index1 = a.index(max1)
    # prediction[istart:iend] = np.expand_dims(result.argmax(axis=1),axis=1)/16
    prediction[istart:iend] = result
prediction = np.asarray(prediction)
resultPerFace=prediction
if mode == "MESH":
    resultPerVertex = np.zeros((len(mModel.vertices)))
    for mVertexIndex, mVertex in enumerate(mModel.vertices):
        umbrella = [prediction[mFaceIndex] for mFaceIndex in mVertex.neighbouringFaceIndices]
        umbrella = np.asarray(umbrella)
        resultPerVertex[mVertexIndex] = np.max(umbrella)
if mode == "PC":
    resultPerVertex = prediction
for i, v in enumerate(mModel.vertices):
    # h=-((resultPerVertex[i] * 240*0.25))
    h = 0
    # s=1.0
    s = 0
    # v=1.0
    v = (resultPerVertex[i])
    r, b, g = hsv2rgb(h, s, v)
    mModel.vertices[i] = mModel.vertices[i]._replace(color=np.asarray([r / 255, g / 255, b / 255]))

modelToSave=modelName + "_" + reshapeFunction + "_pred" + "_cnn_" + "p_" + str(numOfElements)

np.savetxt(rootdir + modelsDir+modelToSave+".csv" , resultPerVertex, delimiter=',',fmt='%.4f')

ffname = rootdir + modelsDir + modelToSave

exportObj(mModel, ffname + ".obj", color=True)

# convert.execute(ffname + ".obj", ffname + ".ply")
# os.remove(ffname + ".obj")




step = (1 / saliencyDivisions)


if showConfMat:
    saliencyGroundTruthPostfix = '_saliencyValues_of_points.csv'
    saliencyGroundTruthPath = rootdir + modelsDir + modelName + saliencyGroundTrouthData
    saliencyGroundTruthData = np.genfromtxt(saliencyGroundTruthPath, delimiter=',')


    pcdgt = o3d.geometry.PointCloud()
    pcdgt.points = o3d.utility.Vector3dVector(np.asarray([f.centroid for f in mModel.faces]))
    pcdgt.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=50))
    voxel_volume = o3d.geometry.VoxelGrid.create_from_point_cloud(pcdgt, voxel_size=0.02)
    # o3d.visualization.draw_geometries([voxel_volume])
    VX = voxel_volume.get_voxels()
    # point_cloud_np = np.asarray(
    #                 [voxel_volume.origin + pt.grid_index * voxel_volume.voxel_size for pt in voxel_volume.get_voxels()])
    # Build correspondence map
    CorrespondenceMap = []
    for faceIndex, face in enumerate(mModel.faces):
        # print(vertexIndex)
        voxelIndex = voxel_volume.get_voxel(face.centroid)
        CorrespondenceMap.append(voxelIndex)
    CorrespondenceMap = np.asarray(CorrespondenceMap)
    CorrespondenceMapSet = {(tuple(row)) for row in CorrespondenceMap}
    CorrespondenceMapList = list(CorrespondenceMapSet)
    CorrespondenceMapDict = {}
    for c in CorrespondenceMapList:
        CorrespondenceMapDict[c] = []
    for faceIndex, face in enumerate(mModel.faces):
        voxelIndex = voxel_volume.get_voxel(face.centroid)
        CorrespondenceMapDict[tuple(voxelIndex)].append(faceIndex)
    vgdats = []
    vpdats = []
    vpoints = []
    for pt in voxel_volume.get_voxels():
        voxPoint = voxel_volume.origin + pt.grid_index * voxel_volume.voxel_size
        relevantFaces = CorrespondenceMapDict[tuple(pt.grid_index)]
        gtPointValue = np.mean(saliencyGroundTruthData[relevantFaces])
        predPointValue = np.mean(resultPerFace[relevantFaces])
        vgdats.append(gtPointValue)
        vpdats.append(predPointValue)
        vpoints.append(voxPoint)
    gtdataVoxelized = np.asarray(vgdats)
    predictionUpdatedVoxelized = np.asarray(vpdats)



    # _true = np.clip((np.floor((gtdataVoxelized/ step))).astype(int), a_min=0,
    #                 a_max=(saliencyDivisions - 1)).astype(
    #     int).tolist()
    # _pred = np.clip((np.floor((predictionUpdatedVoxelized / step))).astype(int), a_min=0,
    #                 a_max=(saliencyDivisions - 1)).astype(int).tolist()

    _true = np.clip((np.floor((saliencyGroundTruthData/ step))).astype(int), a_min=0,
                    a_max=(saliencyDivisions - 1)).astype(
        int).tolist()
    _pred = np.clip((np.floor((resultPerFace / step))).astype(int), a_min=0,
                    a_max=(saliencyDivisions - 1)).astype(int).tolist()

    classes = np.asarray(range(0, saliencyDivisions)).astype(int).tolist()
    normalize = True
    cm = confusion_matrix(_true, _pred,labels=list(range(saliencyDivisions)))
    print(cm)
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    if showTags:
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title='',
               ylabel='True label',
               xlabel='Predicted label')
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center", fontsize=10,
                        color="white" if cm[i, j] > thresh else "black")
    #fig.tight_layout()
    #fig.savefig('./results/cms/' + modelName + '_' + reshapeFunction + '_cnn_' + 'p_' + str(numOfElements),dpi=fig.dpi)

    # importing accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    print('\nAccuracy: {:.2f}\n'.format(accuracy_score(_true, _pred)))

    print('Micro Precision: {:.2f}'.format(precision_score(_true, _pred, average='micro')))
    print('Micro Recall: {:.2f}'.format(recall_score(_true, _pred, average='micro')))
    print('Micro F1-score: {:.2f}\n'.format(f1_score(_true, _pred, average='micro')))

    print('Macro Precision: {:.2f}'.format(precision_score(_true, _pred, average='macro')))
    print('Macro Recall: {:.2f}'.format(recall_score(_true, _pred, average='macro')))
    print('Macro F1-score: {:.2f}\n'.format(f1_score(_true, _pred, average='macro')))

    print('Weighted Precision: {:.2f}'.format(precision_score(_true, _pred, average='weighted')))
    print('Weighted Recall: {:.2f}'.format(recall_score(_true, _pred, average='weighted')))
    print('Weighted F1-score: {:.2f}'.format(f1_score(_true, _pred, average='weighted')))

    from sklearn.metrics import classification_report

    print('\nClassification Report\n')
    print(classification_report(_true, _pred))


    plt.show()





