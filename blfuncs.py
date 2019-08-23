#use with blender 2.8.0
import numpy as np

def newobj(name, location, vertices, edge, polygons):
    '''
    create obj base on local vertice coordination
    '''
    #set mesh name and objname the same
    newmesh = bpy.data.meshes.new(name) 
    newobject = bpy.data.objects.new(name, newmesh)
    # set mesh location: if =[] means local origin is at the world origin
    newobject.location = location 
    bpy.context.scene.collection.objects.link(newobject)
    # create mesh from python data
    newmesh.from_pydata(vertices, edge, polygons)
    newmesh.update(calc_edges=True)
    return newobject

def getFaceNormals(vertices, pols):
    """ 
    Returns normals for each facet of mesh 
    Assuming right hand rule
    Require: pols has to be numpy array --> Need to generalize to polygons with
    diffferent number of vertices
    """
    pols = np.array(pols)
    u = vertices[pols[:,1],:] - vertices[pols[:,0],:]
    v = vertices[pols[:,2],:] - vertices[pols[:,0],:]
    normals = np.cross(u,v)
    norms = np.sqrt(np.sum(normals*normals, axis=1))
    return normals/norms[:, np.newaxis]

def getVertNormals(polygons, faceNorms):
    """ 
    Returns normals for each verts of mesh 
    based on norms of pols
    """
    normVectors = [] #[verticeNO, meansNOrm[n,3]numpyarray]
    for count, pol in enumerate(polygons):
        if count == 0:
            for ver in pol:
                normVectors.append([ver, faceNorms[count]])
        else:
            for ver in pol:
                #check if ver is in normVectors[0] if count >=1
                list0 = [x[0] for x in normVectors]
                if ver in list0: 
                    idx=list0.index(ver)
                    normVectors[idx][1] =  np.vstack(
                        [normVectors[idx][1], faceNorms[count]])
                else:
                    normVectors.append([ver, faceNorms[count]])
    normVectors = sorted(normVectors)

    #mean all normal vectors
    meanarray = []
    for ver in normVectors:
        norm = ver[1]
        if norm.ndim == 2:
            norm = np.mean(norm, axis=0) #some has 1 dim
        meanarray.append(norm)
    return np.array(meanarray)

def to_global(matrix_world, local_coords):
#https://blender.stackexchange.com/questions/6155/how-to-convert-coordinates-from-vertex-to-world-space
    #add an extra column of one
    local_coords = np.c_[local_coords, np.ones(local_coords.shape[0])]
    global_coords = np.dot(matrix_world, local_coords.T)[0:3].T
    return global_coords

def getobjdata(obj, glo):
    '''
    get vertice coords and polygons of obj
    default is in local coord system
    if glo=True -> export vertices in global coord system 
    '''
    #get active object
    objname = obj.name
    objdata = bpy.data.meshes[objname]
    objlocation = bpy.data.objects[objname].location
    #get matrix of polygon and vertices coords
    vertices=[]
    polygons=[]
    for ver in objdata.vertices:
        vertices.append(ver.co)
    vertices = np.array(vertices)
    for pol in objdata.polygons:
        vertindex_list=[]
        for verNo in range(0, len(pol.vertices)):
            vertindex_list.append(pol.vertices[verNo])
        polygons.append(vertindex_list)
    if glo==True:
#https://b3d.interplanety.org/en/how-to-get-global-vertex-coordinates/
        #vertices = obj.matrix_world @ vertices
        vertices = to_global(obj.matrix_world, vertices)
        #print(vertices)

    return vertices, polygons, objlocation #return nparray, list, list 


def drawsemiArrows(name, location, originVerts, endVerts):
    '''
    draw semi arrow (edges) from 2 arrays of begin and end Verts 
    '''
    arr_vertices = np.vstack([originVerts, endVerts])
    arr_edge = [] 
    no_ver = len(originVerts)
    for c in range(0, no_ver):
        arr_edge.append([c, no_ver + c])
    objdict[name] = newobj(name, location, arr_vertices,
                                   arr_edge, [])
    return


def drawsemiArrowsPol(name, location, vertices, polygons):
    '''
    draw arrows from middle of pols along their norm vectors
    '''
    #draw arrow to check norm vector of pol:
    middleVertices = np.zeros((len(polygons),3))
    faceNorms = getFaceNormals(vertices, polygons) 
    for count, pol in enumerate(polygons):
        polvertices = []
        for ver in pol:
            polvertices.append(vertices[ver])
        polvertice = np.array(polvertices)
        middleVer = np.mean(polvertices, axis=0)
        middleVertices[count] = middleVer

    new_vertices = middleVertices + faceNorms
    obj = drawsemiArrows(name, location, middleVertices, new_vertices)

    return obj


def drawsemiArrowsVer(name, location, vertices, polygons):
    '''
    draw arrows from extruding vertices along their mean norm vectors
    which calculated from norm vector of surrounding faces
    '''

    faceNorms = getFaceNormals(vertices, polygons) #normal vector for faces
    verNorms = getVertNormals(polygons, faceNorms) 

    new_vertices=np.zeros((len(vertices),3))
    for i in range (0, len(vertices)):
        new_vertices[i] = vertices[i] + verNorms[i] 

    obj = drawsemiArrows(name, location, vertices, new_vertices)

    return obj

def extrudeVertsToplane(name, fromobj, magnitude ):
    '''
    create new object by extruding vertices along their norm vector of vertices
    '''
    vertices, polygons, location = getobjdata(fromobj, glo=False)
    faceNorms = getFaceNormals(vertices, polygons) #normal vector for faces
    verNorms = getVertNormals(polygons, faceNorms) 
    new_vertices = vertices + verNorms*magnitude
    obj = newobj(name, location, new_vertices, [], polygons)
    return obj
