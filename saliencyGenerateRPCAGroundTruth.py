import tensorflow as tf
import numpy as np
import random
import gzip
import tarfile
import pickle
import os
from six.moves import urllib
from commonReadOBJPointCloud import *
import scipy.io
import sklearn as sk
from scipy.spatial.distance import pdist, squareform
import common.hilbertcurve.hilbertcurve.hilbertcurve as hb
import pickle
import glob
import scipy.sparse as sp
from robust_pca import R_pca
from scipy.spatial.distance import directed_hausdorff
from scipy.sparse.linalg import inv
import matplotlib.pyplot as plt
import time
from numba import jit, cuda

def pointCloudToRPCASaliency(Vertices,pointcloudnn,rpcaneighbours,presimplification):
    mModel = NumpyArrayToPointCloudStructure(Vertices, nn=pointcloudnn, simplify=presimplification)
    V, inds = computePointCloudNormals(mModel, pointcloudnn)
    Normals = []
    eigenvals = []
    print(80 * "=")
    print('Spectral saliency')
    print(80 * "=")
    iLen = len(mModel.vertices)
    for v_ind, f in enumerate(mModel.vertices):
        if v_ind % 2000 == 0:
            print('Extract patch information : ' + str(
                np.round((100 * v_ind / iLen), decimals=2)) + ' ' + '%')
        NormalsLine = np.empty(shape=[0, 3])
        patchFaces, rings = neighboursByVertex(mModel, v_ind, rpcaneighbours)
        for j in patchFaces:
            NormalsLine = np.append(NormalsLine, [mModel.vertices[j].normal], axis=0)
        nn = np.asarray(NormalsLine)
        conv1 = np.matmul(np.transpose(nn), nn)
        w, v = LA.eig(conv1)
        val = 1 / np.linalg.norm(w)
        eigenvals.append(val)
        NormalsLine = NormalsLine.ravel()
        Normals.append(NormalsLine)
    print(80 * "=")
    print('Geometric saliency')
    print(80 * "=")
    Normals = np.asarray(Normals)
    lmbda = 1 / np.sqrt(np.max(Normals.shape))
    mu = 10 * lmbda
    rpca = R_pca(Normals, lmbda=lmbda, mu=mu)
    LD, SD = rpca.fit(max_iter=700, iter_print=100, tol=1E-2)
    print("RPCA Shape:" + str(np.shape(Normals)) + "," + "Matrix Rank:" + str(
        np.linalg.matrix_rank(Normals)))
    print("LD Shape:" + str(np.shape(LD)) + "," + "Matrix Rank:" + str(np.linalg.matrix_rank(LD)) + " Max Val : " + str(
        np.max(LD)))
    print("SD Shape:" + str(np.shape(SD)) + "," + "Matrix Rank:" + str(np.linalg.matrix_rank(SD)) + " Max Val : " + str(
        np.max(SD)))

    CurvatureComponent = np.asarray(eigenvals)
    RPCAComponent = np.sum(np.abs(SD) ** 2, axis=-1) ** (1. / 2)
    print(np.shape(RPCAComponent))
    print(80 * "=")
    print('Combine')
    print(80 * "=")
    S1 = (RPCAComponent - RPCAComponent.min()) / (RPCAComponent.max() - RPCAComponent.min())
    E1 = (CurvatureComponent - CurvatureComponent.min()) / (CurvatureComponent.max() - CurvatureComponent.min())
    saliencyCombined = (S1 + E1) / 2
    saliencyCombined = saliencyCombined / np.max(saliencyCombined)

    return mModel,saliencyCombined


def pointCloudToRPCASaliencyWithReflectance(Vertices,Reflectance,pointcloudnn,rpcaneighbours,presimplification):
    mModel = NumpyArrayToPointCloudStructure(Vertices, nn=pointcloudnn, simplify=presimplification)
    V, inds = computePointCloudNormals(mModel, pointcloudnn)
    Normals = []
    Reflectances=[]
    eigenvals = []
    print(80 * "=")
    print('Spectral saliency')
    print(80 * "=")
    iLen = len(mModel.vertices)



    for v_ind, f in enumerate(mModel.vertices):
        if v_ind % 2000 == 0:
            print('Extract patch information : ' + str(
                np.round((100 * v_ind / iLen), decimals=2)) + ' ' + '%')
        NormalsLine = np.empty(shape=[0, 3])

        ReflectanceLine=[]
        patchFaces, rings = neighboursByVertex(mModel, v_ind, rpcaneighbours)
        for j in patchFaces:
            NormalsLine = np.append(NormalsLine, [mModel.vertices[j].normal], axis=0)
            ReflectanceLine.append(Reflectance[j])

        ReflectanceLine=np.asarray(ReflectanceLine)
        nn = np.asarray(NormalsLine)
        conv1 = np.matmul(np.transpose(nn), nn)
        w, v = LA.eig(conv1)
        val = 1 / np.linalg.norm(w)
        eigenvals.append(val)
        NormalsLine = NormalsLine.ravel()
        Normals.append(NormalsLine)
        Reflectances.append(ReflectanceLine)

    print(80 * "=")
    print('Reflectance saliency')
    print(80 * "=")
    Reflectances = np.asarray(Reflectances)
    lmbda = 1 / np.sqrt(np.max(Reflectances.shape))
    mu = 10 * lmbda
    rpca = R_pca(Reflectances, lmbda=lmbda, mu=mu)
    LD, SD = rpca.fit(max_iter=700, iter_print=100, tol=1E-2)
    print("RPCA Shape:" + str(np.shape(Reflectances)) + "," + "Matrix Rank:" + str(
        np.linalg.matrix_rank(Reflectances)))
    print("LD Shape:" + str(np.shape(LD)) + "," + "Matrix Rank:" + str(np.linalg.matrix_rank(LD)) + " Max Val : " + str(
        np.max(LD)))
    print("SD Shape:" + str(np.shape(SD)) + "," + "Matrix Rank:" + str(np.linalg.matrix_rank(SD)) + " Max Val : " + str(
        np.max(SD)))
    RPCAReflectanceComponent = np.sum(np.abs(SD) ** 2, axis=-1) ** (1. / 2)
    R1 = (RPCAReflectanceComponent - RPCAReflectanceComponent.min()) / (
                RPCAReflectanceComponent.max() - RPCAReflectanceComponent.min())
    print(np.shape(RPCAReflectanceComponent))




    print(80 * "=")
    print('Geometric saliency')
    print(80 * "=")
    Normals = np.asarray(Normals)
    lmbda = 1 / np.sqrt(np.max(Normals.shape))
    mu = 10 * lmbda
    rpca = R_pca(Normals, lmbda=lmbda, mu=mu)
    LD, SD = rpca.fit(max_iter=700, iter_print=100, tol=1E-2)
    print("RPCA Shape:" + str(np.shape(Normals)) + "," + "Matrix Rank:" + str(
        np.linalg.matrix_rank(Normals)))
    print("LD Shape:" + str(np.shape(LD)) + "," + "Matrix Rank:" + str(np.linalg.matrix_rank(LD)) + " Max Val : " + str(
        np.max(LD)))
    print("SD Shape:" + str(np.shape(SD)) + "," + "Matrix Rank:" + str(np.linalg.matrix_rank(SD)) + " Max Val : " + str(
        np.max(SD)))

    CurvatureComponent = np.asarray(eigenvals)
    E1 = (CurvatureComponent - CurvatureComponent.min()) / (CurvatureComponent.max() - CurvatureComponent.min())

    RPCAComponent = np.sum(np.abs(SD) ** 2, axis=-1) ** (1. / 2)
    S1 = (RPCAComponent - RPCAComponent.min()) / (RPCAComponent.max() - RPCAComponent.min())
    print(np.shape(RPCAComponent))







    return mModel,S1,E1,R1

# # function optimized to run on gpu
# @jit(target ="cuda")
def meshRPCASaliency(mModel):
    Normals = []
    eigenvals = []
    print(80 * "=")
    print('Spectral saliency')
    print(80 * "=")
    iLen = len(mModel.faces)
    for f_ind, f in enumerate(mModel.faces):
        if f_ind % 2000 == 0:
            print('Extract patch information : ' + str(
                np.round((100 * f_ind / iLen), decimals=2)) + ' ' + '%')
        NormalsLine = np.empty(shape=[0, 3])
        patchFaces, rings = neighboursByFace(mModel, f_ind, rpcaneighbours)
        for j in patchFaces:
            NormalsLine = np.append(NormalsLine, [mModel.faces[j].faceNormal], axis=0)
        nn = np.asarray(NormalsLine)
        conv1 = np.matmul(np.transpose(nn), nn)
        w, v = LA.eig(conv1)
        val = 1 / np.linalg.norm(w)
        eigenvals.append(val)
        NormalsLine = NormalsLine.ravel()
        Normals.append(NormalsLine)
    print(80 * "=")
    print('Geometric saliency')
    print(80 * "=")
    Normals = np.asarray(Normals)
    lmbda = 1 / np.sqrt(np.max(Normals.shape))
    mu = 10 * lmbda
    rpca = R_pca(Normals, lmbda=lmbda, mu=mu)
    LD, SD = rpca.fit(max_iter=700, iter_print=100, tol=1E-2)
    print("RPCA Shape:" + str(np.shape(Normals)) + "," + "Matrix Rank:" + str(
        np.linalg.matrix_rank(Normals)))
    print("LD Shape:" + str(np.shape(LD)) + "," + "Matrix Rank:" + str(np.linalg.matrix_rank(LD)) + " Max Val : " + str(
        np.max(LD)))
    print("SD Shape:" + str(np.shape(SD)) + "," + "Matrix Rank:" + str(np.linalg.matrix_rank(SD)) + " Max Val : " + str(
        np.max(SD)))
    CurvatureComponent = np.asarray(eigenvals)
    RPCAComponent = np.sum(np.abs(SD) ** 2, axis=-1) ** (1. / 2)
    print(np.shape(RPCAComponent))
    print(80 * "=")
    print('Combine')
    print(80 * "=")
    S1 = (RPCAComponent - RPCAComponent.min()) / (RPCAComponent.max() - RPCAComponent.min())
    E1 = (CurvatureComponent - CurvatureComponent.min()) / (CurvatureComponent.max() - CurvatureComponent.min())
    saliencyCombined = (S1 + E1) / 2
    saliencyCombined = saliencyCombined / np.max(saliencyCombined)
    return saliencyCombined



if True:
    # ========= Generic configurations =========#
    print(80 * "=")
    print('Initialize')
    print(80 * "=")
    rootdir='/home/stavros/Workspace/Mesh-Saliency-Extraction-Compression-Simplification/saliency/'
    modelsDir='data/'
    modelFilename='head.obj'
    fullModelPath=rootdir+modelsDir+modelFilename
    fpath=fullModelPath.split(sep=".")[0]
    patchSide=32
    numOfElements = patchSide * patchSide
    numberOfClasses=4
    saliencyDivisions=64
    useGuided = False
    doRotate = True
    doReadOBJ = True
    rpcaneighbours=60
    pointcloudnn=8
    mode = "MESH"
    patchSizeGuided = numOfElements
    # ========= Read models =========#
    print(80 * "=")
    print('Read model data')
    print(80 * "=")
    (path, file) = os.path.split(fullModelPath)
    filename, file_extension = os.path.splitext(file)
    modelName=filename
    mModelSrc = rootdir +modelsDir+ modelName + '.obj'
    print(modelName)
    t = time.time()
    presimplification=None



    if mode == "MESH":
        saliencyGroundTrouthData = '_saliencyValues_of_centroids.csv'
        mModel = loadObj(mModelSrc)
        keyPredict = 'model_mesh' + modelName
        updateGeometryAttibutes(mModel, useGuided=useGuided, numOfFacesForGuided=patchSizeGuided, computeDeltas=False,
                                computeAdjacency=False, computeVertexNormals=False)
        saliencyCombined=meshRPCASaliency(mModel)
        trm = time.time() - t
        print(80 * "=")
        print("Total time : " + str(trm))
        print(80 * "=")
        saliencyPerFace=saliencyCombined
        saliencyPerVertex = np.zeros((len(mModel.vertices)))
        for mVertexIndex, mVertex in enumerate(mModel.vertices):
            umbrella = [saliencyPerFace[mFaceIndex] for mFaceIndex in mVertex.neighbouringFaceIndices]
            umbrella = np.asarray(umbrella)
            saliencyPerVertex[mVertexIndex] = np.max(umbrella)
        np.savetxt(fpath + saliencyGroundTrouthData, saliencyPerFace, delimiter=',', fmt='%10.3f')
        # --- color models ------
        for i, v in enumerate(mModel.vertices):
            # h=-((resultPerVertex[i] * 240*0.25))
            h = 0
            # s=1.0
            s = 0
            # v=1.0
            v = saliencyPerVertex[i]
            r, b, g = hsv2rgb(h, s, v)
            mModel.vertices[i] = mModel.vertices[i]._replace(color=np.asarray([r, g, b]))
        exportObj(mModel, fpath + "_gt_test" + ".obj", color=True)


    if mode == "PC":
        saliencyGroundTrouthData = '_saliencyValues_of_points.csv'
        # mModel = loadObjPC(mModelSrc, nn=pointcloudnn, simplify=presimplification)
        # V, inds = computePointCloudNormals(mModel, pointcloudnn)
        Vertices=loadObjPointCloudFileToNumpyArray(mModelSrc, nn=pointcloudnn, simplify=presimplification)
        mModel,saliencyCombined=pointCloudToRPCASaliency(Vertices)
        saliencyPerVertex=saliencyCombined
        np.savetxt(fpath + saliencyGroundTrouthData, saliencyPerVertex, delimiter=',', fmt='%10.3f')
        # --- color models ------
        for i, v in enumerate(mModel.vertices):
            # h=-((resultPerVertex[i] * 240*0.25))
            h = 0
            # s=1.0
            s = 0
            # v=1.0
            v = saliencyPerVertex[i]
            r, b, g = hsv2rgb(h, s, v)
            mModel.vertices[i] = mModel.vertices[i]._replace(color=np.asarray([r, g, b]))
        exportObj(mModel, fpath + "_gt_test" + ".obj", color=True)






