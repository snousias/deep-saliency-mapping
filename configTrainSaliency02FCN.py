import numpy as np
import glob
import os
import hilbertcurve.hilbertcurve.hilbertcurve as hb
rootdir = './'
modelsDir='data/'
sessionsDir='models/'


patchSide=32
numEpochs=5
numOfElements = patchSide * patchSide
numberOfClasses=16
selectedModel=1
saliencyDivisions=64
pointcloudnn=8
rate=0.0
# Training params
final_iter = 25000
# Assign the batch value
batch_size = 100
# 20% of the data will automatically be used for validation
validation_size = 0.05
img_size = patchSide
num_channels = 3
num_classes = numberOfClasses
type="continuous"
mode="MESH"
reshapeFunction='hilbert'
batchNormalized=''
target = np.asarray([0.0, 1.0, 0.0]) #Rotation
useGuided=False
presimplification =None
keyword="saliency_fcn"








# =========Derived=====================================================================================================
if batchNormalized=='normalized':
    saliencyGroundTrouthData='_saliencyValues_of_batch_centroids.csv'
if batchNormalized=='':
    saliencyGroundTrouthData = '_saliencyValues_of_centroids.csv'
reshapeFunction=reshapeFunction+batchNormalized
patchSizeGuided = numOfElements
if mode=="PC":
    keyTrain = '_'+keyword + '_'+str(patchSide)+'_models_point_cloud' + 'reshaping_' + reshapeFunction + type+'_' +  str(numberOfClasses)
if mode=="MESH":
    keyTrain = '_'+keyword + '_'+str(patchSide)+'_models_mesh' + 'reshaping_' + reshapeFunction + type+'_' + str(numberOfClasses)





#Read Models============================================================================================================
trainSetIndices=[selectedModel]
g=glob.glob(rootdir+modelsDir+'*.obj')
trainModels=[]
for i in range(0,len(g)):
    (path, file)  = os.path.split(g[i])
    filename, file_extension = os.path.splitext(file)
    filenameParts=filename.split("_")
    trainModels.append(filenameParts[0])
trainModels = (list(set(trainModels)))
trainModels.sort()

#Hilbert Curve==========================================================================================================
p = patchSide
N = 2
hilbert_curve = hb.HilbertCurve(p, N)
I2HC=np.empty((p*p,2))
HC2I=np.empty((p,p))
hCoords=[]
for ii in range(p*p):
    h=hilbert_curve.coordinates_from_distance(ii)
    hCoords.append(h)
    I2HC[ii,:]=h
    HC2I[h[0],h[1]]=ii
I2HC=I2HC.astype(int)
HC2I=HC2I.astype(int)


groundTruthKeyword='_saliencyValues_of_centroids'

# saliencyValue = np.genfromtxt(saliencyValuePath, delimiter=',')
# #saliencyValue=saliencyValue/np.max(saliencyValue)
# print('Saliency ground truth data :', saliencyValuePath)
# if type == 'continuous':
#     for s in saliencyValue:
#         saliencyValues.append(np.asarray(s))
# if type == 'discrete':
#     for s in saliencyValue:
#         v = int((num_classes - 1) * s)
#         saliencyValues.append(v)

# valuePerFaceIn = saliencyValue
# valuePerVertexOut = np.zeros((len(mModel.vertices)))
# for mVertexIndex, mVertex in enumerate(mModel.vertices):
#     umbrella = [valuePerFaceIn[mFaceIndex] for mFaceIndex in mVertex.neighbouringFaceIndices]
#     umbrella = np.asarray(umbrella)
#     valuePerVertexOut[mVertexIndex] = np.max(umbrella)
# valuePerVertexIn = np.zeros((len(mModel.vertices)))