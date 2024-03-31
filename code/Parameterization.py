import os
import polyscope as ps
import polyscope.imgui as psim
import numpy as np
from DAFunctions import load_off_file, compute_laplacian, compute_edge_list, compute_boundary_embedding, compute_tutte_embedding


def visualize_boundary_Edges(ps_mesh, vertices, boundEdges):
    boundVertices = vertices[boundEdges]

    boundVertices = boundVertices.reshape(2 * boundVertices.shape[0], 3)
    curveNetIndices = np.arange(0, boundVertices.shape[0])
    curveNetIndices = curveNetIndices.reshape(int(len(curveNetIndices) / 2), 2)
    ps_net = ps.register_curve_network("boundary edges", boundVertices, curveNetIndices)

    return ps_net


if __name__ == '__main__':
    ps.init()

    vertices, faces = load_off_file(os.path.join('data', 'param', 'cathead.off'))

    ps_mesh = ps.register_surface_mesh("Input Mesh", vertices, faces)
    halfedges, edges, edgeBoundMask, boundVertices, EH, EF = compute_edge_list(vertices, faces, sortBoundary=True)
    L, vorAreas, d0, W = compute_laplacian(vertices, faces, edges, edgeBoundMask, EF)

    r = 1.0

    boundUV = compute_boundary_embedding(vertices, boundVertices, r)
    UV = compute_tutte_embedding(vertices, d0, W, boundVertices, boundUV)

    ps.register_surface_mesh("UV Mesh", np.column_stack((UV[:, 0], np.zeros(UV.shape[0]), UV[:, 1])), faces)

    ps_mesh.add_parameterization_quantity("UV Mapping", UV, coords_type='world')

    ps_net = visualize_boundary_Edges(ps_mesh, vertices,  edges[edgeBoundMask == 1, :])

    ps.show()
