import numpy as np
import scipy.linalg
from scipy.sparse import csc_matrix, csr_matrix, coo_matrix, linalg, bmat, diags
from scipy.sparse.linalg import spsolve, lsqr

def accumarray(indices, values):
    output = np.zeros((np.max(indices) + 1), dtype=values.dtype)
    indFlat = indices.flatten()
    valFlat = values.flatten()
    
    # for index in range(indFlat.shape[0]):
    #     output[indFlat[index]] += valFlat[index]
    np.add.at(output, indFlat, valFlat)

    return output


def load_off_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Parse the vertices and faces from the OFF file
    num_vertices, num_faces, _ = map(int, lines[1].split())

    vertices = np.array([list(map(float, line.split())) for line in lines[2:2 + num_vertices]])
    faces = np.array([list(map(int, line.split()))[1:] for line in lines[2 + num_vertices:]])

    return vertices, faces


def compute_areas_normals(vertices, faces):
    face_vertices = vertices[faces]

    # Compute vectors on the face
    vectors1 = face_vertices[:, 1, :] - face_vertices[:, 0, :]
    vectors2 = face_vertices[:, 2, :] - face_vertices[:, 0, :]

    # Compute face normals using cross product
    normals = np.cross(vectors1, vectors2)
    faceAreas = 0.5 * np.linalg.norm(normals, axis=1)

    normals /= (2.0 * faceAreas[:, np.newaxis])
    return normals, faceAreas


def visualize_boundary_Edges(ps_mesh, vertices, boundEdges):
    boundVertices = vertices[boundEdges]

    boundVertices = boundVertices.reshape(2 * boundVertices.shape[0], 3)
    curveNetIndices = np.arange(0, boundVertices.shape[0])
    curveNetIndices = curveNetIndices.reshape(int(len(curveNetIndices) / 2), 2)
    ps_net = ps.register_curve_network("boundary edges", boundVertices, curveNetIndices)

    return ps_net


def createEH(edges, halfedges):
    # Create dictionaries to map halfedges to their indices
    halfedges_dict = {(v1, v2): i for i, (v1, v2) in enumerate(halfedges)}
    # reversed_halfedges_dict = {(v2, v1): i for i, (v1, v2) in enumerate(halfedges)}

    EH = np.zeros((len(edges), 2), dtype=int)

    for i, (v1, v2) in enumerate(edges):
        # Check if the halfedge exists in the original order
        if (v1, v2) in halfedges_dict:
            EH[i, 0] = halfedges_dict[(v1, v2)]
        # Check if the halfedge exists in the reversed order
        if (v2, v1) in halfedges_dict:
            EH[i, 1] = halfedges_dict[(v2, v1)]

    return EH


def compute_edge_list(vertices, faces, sortBoundary=False):
    halfedges = np.empty((3 * faces.shape[0], 2))
    for face in range(faces.shape[0]):
        for j in range(3):
            halfedges[3 * face + j, :] = [faces[face, j], faces[face, (j + 1) % 3]]

    edges, firstOccurence, numOccurences = np.unique(np.sort(halfedges, axis=1), axis=0, return_index=True,
                                                     return_counts=True)
    edges = halfedges[np.sort(firstOccurence)]
    edges = edges.astype(int)
    halfedgeBoundaryMask = np.zeros(halfedges.shape[0])
    halfedgeBoundaryMask[firstOccurence] = 2 - numOccurences
    edgeBoundMask = halfedgeBoundaryMask[np.sort(firstOccurence)]

    boundEdges = edges[edgeBoundMask == 1, :]
    boundVertices = np.unique(boundEdges).flatten()

    # EH = [np.where(np.sort(halfedges, axis=1) == edge)[0] for edge in edges]
    # EF = []

    EH = createEH(edges, halfedges)
    EF = np.column_stack((EH[:, 0] // 3, (EH[:, 0] + 2) % 3, EH[:, 1] // 3, (EH[:, 1] + 2) % 3))

    if (sortBoundary):
        loop_order = []
        loopEdges = boundEdges.tolist()
        current_node = boundVertices[0]  # Start from any node
        visited = set()
        while True:
            loop_order.append(current_node)
            visited.add(current_node)
            next_nodes = [node for edge in loopEdges for node in edge if
                          current_node in edge and node != current_node and node not in visited]
            if not next_nodes:
                break
            next_node = next_nodes[0]
            loopEdges = [edge for edge in loopEdges if
                          edge != (current_node, next_node) and edge != (next_node, current_node)]
            current_node = next_node
            current_node = next_node

        boundVertices = np.array(loop_order)

    return halfedges, edges, edgeBoundMask, boundVertices, EH, EF




def compute_angle_defect(vertices, faces, boundVertices):
    edges = vertices[faces[:, [1, 2, 0]]] - vertices[faces[:, [0, 1, 2]]]
    lengths = np.linalg.norm(edges, axis=2)
    
    dot_products = np.sum(edges * np.roll(edges, -1, axis=1), axis=2)
    angles = np.arccos(dot_products / (lengths * np.roll(lengths, -1, axis=1)))
    angles = np.pi - np.roll(angles, 1, axis=1)
    
    G_inner = 2 * np.pi - accumarray(faces, angles)
    G_inner[boundVertices] -= np.pi

    return G_inner


def compute_mean_curvature_normal(vertices, faces, faceNormals, L, vorAreas):
    Lv = L @ vertices
    vertexNormals = np.zeros_like(vertices)

    for v_idx in range(vertices.shape[0]):
        neighbor_faces = np.where(faces == v_idx)[0]
        normal_sum = np.sum(faceNormals[neighbor_faces], axis=0)
        vertexNormals[v_idx] = normal_sum / np.linalg.norm(normal_sum)
    
    MCNormal = Lv / (2 * vorAreas[:, np.newaxis]) 
    
    MC = np.linalg.norm(MCNormal, axis=1)
    
    dot_product = np.sum(MCNormal * vertexNormals, axis=1)
    MC *= np.sign(dot_product)
    
    return MCNormal, MC, vertexNormals


def compute_laplacian(vertices, faces, edges, edgeBoundMask, EF):
    __, face_areas = compute_areas_normals(vertices, faces)
    vorAreas = accumarray(faces, np.repeat(face_areas, 3)) / 3

    num_edges = len(edges)
    num_vertices = len(vertices)

    left_faces = faces[EF[:,0]]
    right_faces = faces[EF[:,2]]
    j_indexes = EF[:,1]
    l_indexes = EF[:,3]

    W_values = np.zeros(num_edges)
    computed_cot = {}

    for i, edge in enumerate(edges):
        j = left_faces[i, j_indexes[i]]

        if (i, j) not in computed_cot:
            v1 = vertices[edge[0]] - vertices[j]
            v2 = vertices[edge[1]] - vertices[j]
            dot_prod = np.dot(v1, v2)
            cross_product_norm = np.linalg.norm(np.cross(v1, v2))
            cot_j = 0.5 * dot_prod / cross_product_norm
            computed_cot[(i, j)] = cot_j
        else:
            cot_j = computed_cot[(i, j)]

        W_values[i] += cot_j

        if not edgeBoundMask[i]:
            l = right_faces[i, l_indexes[i]]
            
            if (i, l) not in computed_cot:
                v1 = vertices[edge[0]] - vertices[l]
                v2 = vertices[edge[1]] - vertices[l]
                dot_prod = np.dot(v1, v2)
                cross_product_norm = np.linalg.norm(np.cross(v1, v2))
                cot_l = 0.5 * dot_prod / cross_product_norm
                computed_cot[(i, l)] = cot_l
            else:
                cot_l = computed_cot[(i, l)]
            W_values[i] += cot_l

    W = diags(W_values, shape=(num_edges, num_edges), format='csc')

    column = edges.flatten()
    data = np.tile(np.array([-1,1]), num_edges)
    row = np.repeat(np.arange(num_edges), 2)
    d0 = csr_matrix((data, (row, column)), shape=(num_edges, num_vertices))
    
    L = d0.T @ W @ d0
    return L, vorAreas, d0, W


def mean_curvature_flow(faces, boundVertices, currVertices, L, vorAreas, flowRate, isExplicit):
    temp_vertices = currVertices.copy()
    
    if isExplicit:
        Lv = L @ currVertices

        flow_step = flowRate * (Lv / vorAreas[:, np.newaxis])
        currVertices -= flow_step

        currVertices[boundVertices] = temp_vertices[boundVertices]
    
    else:
        M = diags(vorAreas, shape=(len(currVertices), len(currVertices)), format='csc')

        lhs_coefficient = M + flowRate * L
        rhs = M @ currVertices

        solver = linalg.splu(lhs_coefficient)
        currVertices = solver.solve(rhs)
        currVertices[boundVertices] = temp_vertices[boundVertices]
    
    
    if boundVertices.size == 0:
        __, face_areas = compute_areas_normals(currVertices, faces)
        
        current_surface_area = np.sum(face_areas)
        original_surface_area = np.sum(vorAreas)
        
        scaling_factor = original_surface_area / current_surface_area
        currVertices *= np.sqrt(scaling_factor)
        
    return currVertices
    
    

def compute_boundary_embedding(vertices, boundVertices, r):
    boundary = vertices[boundVertices].copy()
    diff = np.linalg.norm(np.roll(boundary, -1, axis=0) - boundary, axis=1)
    
    psi = np.cumsum(diff)
    psi *= 2 * np.pi / np.sum(diff)
    
    boundUV = r * np.array([np.cos(psi), np.sin(psi)]).T
    return boundUV


def compute_tutte_embedding(vertices, d0, W, boundVertices, boundUV):
    inner_vertices = [index for index, _ in enumerate(vertices) if index not in boundVertices]
    
    d0_I = d0[:, inner_vertices]
    d0_B = d0[:, boundVertices]
    
    A = d0_I.T @ W @ d0_I
    b = -d0_I.T @ W @ d0_B @ boundUV
    
    x = lsqr(A, b[:,0], atol=1e-16, btol=1e-16)[0]
    y = lsqr(A, b[:,1], atol=1e-16, btol=1e-16)[0]
    innerUV = np.column_stack((x, y))
    
    UV = np.zeros((vertices.shape[0], 2))
    UV[boundVertices] = boundUV
    UV[inner_vertices] = innerUV
    
    return UV