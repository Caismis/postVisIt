import numpy as np
import pandas as pd
import os
import vtk
from vtk.util.numpy_support import vtk_to_numpy as vtn
from scipy.interpolate import NearestNDInterpolator

#user input
target = 'xyz.xyz'
variables = ('temp_mean', 'temp_rms')

#main function

def excel_extract(filepath, sheet_names, data_columns):
    dfset = [pd.read_excel(filepath, sheet_name=name) for name in sheet_names]
    return [[np.array(df[column]) for column in data_columns] for df in dfset]

def init(vtkarray, scalarlist):
    datafn = []
    for i in scalarlist:
        func = NearestNDInterpolator((vtkarray[0]), vtkarray[1][i])
        datafn.append(func)
    print('All Data Loaded.')
    return datafn

def allscalar(vtkarray, scalarlist):
    scalars = np.array([vtkarray[1][i] for i in scalarlist]).T
    return np.c_[vtkarray[0], scalars].T

def scatterslice(scalars, z, tol=2e-3):
    return scalars.T[(scalars[2] < z + tol) & (scalars[2] > z - tol)].T

def vtk_loader(path, show_scalarname=False):
    files = os.listdir(path)
    first_file = True
    for file in files:
        if file[-3:] == 'vtk':
            reader = vtk.vtkStructuredGridReader()
            reader.SetFileName(path + file)
            reader.ReadAllScalarsOn()
            reader.Update()
            vtkdata = reader.GetOutput()
            scalar_names = [reader.GetScalarsNameInFile(i) for i in range(reader.GetNumberOfScalarsInFile())]
            scalar_num = len(scalar_names)
            scalarset = [vtn(vtkdata.GetPointData().GetArray(scalar)) for scalar in scalar_names]
            xyz = vtn(vtkdata.GetPoints().GetData())
            if first_file:
                if show_scalarname:
                    for i in range(scalar_num):
                        print('index= ' + str(i) + ' ' + scalar_names[i])
                great_xyz = xyz
                great_scalars = scalarset
                first_file = False
            else:
                great_xyz = np.append(great_xyz, xyz, axis=0)
                for i in range(len(scalarset)):
                    great_scalars[i] = np.append(great_scalars[i], scalarset[i], axis=0)
    return (great_xyz, great_scalars)

#user function
def radialmean(func, radius, z, points=80):
    radialdata = np.empty(points)
    angs = np.linspace(0, 2*np.pi, points, endpoint=False)
    x = radius*np.cos(angs)
    y = radius*np.sin(angs)
    for i in range(points):
        radialdata[i] = func(x[i], y[i], z)
    return np.mean(radialdata)

def radial_vel(x, y, u, v):
    radius = np.sqrt(x**2 + y**2)
    ang_sin = y/max(radius, 1e-6)
    ang_cos = x/max(radius, 1e-6)
    return v*ang_sin + u*ang_cos

def swirl_vel(x, y, u, v):
    radius = np.sqrt(x**2 + y**2)
    ang_sin = y/max(radius, 1e-6)
    ang_cos = x/max(radius, 1e-6)
    return v*ang_cos - u*ang_sin

def mean_radial(funcu, funcv, radius, z, points=80):
    radialdata = np.empty(points)
    angs = np.linspace(0, 2*np.pi, points, endpoint=False)
    x = radius*np.cos(angs)
    y = radius*np.sin(angs)
    for i in range(points):
        radialdata[i] = radial_vel(x[i], y[i], funcu(x[i], y[i], z), funcv(x[i], y[i], z))
    return np.mean(radialdata)

def mean_swirl(funcu, funcv, radius, z, points=80):
    radialdata = np.empty(points)
    angs = np.linspace(0, 2*np.pi, points, endpoint=False)
    x = radius*np.cos(angs)
    y = radius*np.sin(angs)
    for i in range(points):
        radialdata[i] = swirl_vel(x[i], y[i], funcu(x[i], y[i], z), funcv(x[i], y[i], z))
    return np.mean(radialdata)

def radius_swirl(funcu, funcv, endx, z, points=100, rpoints=80):
    lineout = np.empty(points)
    lineint = np.linspace(0, endx, points)
    for i in range(points):
        lineout[i] = mean_swirl(funcu, funcv, lineint[i], z, rpoints)
    return (lineint, lineout)

def radius_radial(funcu, funcv, endx, z, points=100, rpoints=80):
    lineout = np.empty(points)
    lineint = np.linspace(0, endx, points)
    for i in range(points):
        lineout[i] = mean_radial(funcu, funcv, lineint[i], z, rpoints)
    return (lineint, lineout)

def radiusdata(func, endx, z, points=100, rpoints=80):
    lineout = np.empty(points)
    lineint = np.linspace(0, endx, points)
    for i in range(points):
        lineout[i] = radialmean(func, lineint[i], z, rpoints)
    return (lineint, lineout)

def axialmean(func, endz, points=100):
    lineout = np.empty(points)
    lineint = np.linspace(0, endz, points)
    for i in range(points):
        lineout[i] = func(0, 0, lineint[i])
    return (lineint, lineout)

def rdatacollect(setdata, xds, zds):
    datas = []
    for i in range(len(zds)):
        datarow = []
        write_x = True
        for j in range(len(setdata)):
            the_mean = radiusdata(setdata[j], xds[i], zds[i], 100)
            if write_x:
                datarow.append(the_mean[0])
                datarow.append(the_mean[1])
                write_x = False
            else:
                datarow.append(the_mean[1])
        datas.append(datarow)
    return datas

def adatacollect(setdata, zd):
    datarow = []
    write_z = True
    for i in range(len(setdata)):
        the_mean = axialmean(setdata[i], zd, 100)
        if write_z:
            datarow.append(the_mean[0])
            datarow.append(the_mean[1])
            write_z = False
        else:
            datarow.append(the_mean[1])
    return datarow