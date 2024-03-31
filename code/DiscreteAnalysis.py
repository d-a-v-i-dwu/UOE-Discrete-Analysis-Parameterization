import os
import polyscope as ps
import numpy as np
from DAFunctions import load_off_file, compute_areas_normals, compute_laplacian, compute_mean_curvature_normal, compute_edge_list, compute_angle_defect

if __name__ == '__main__':
    ps.init()

    vertices, faces = load_off_file(os.path.join('data', 'bunny.off'))

    ps_mesh = ps.register_surface_mesh("Input Mesh", vertices, faces)

    halfedges, edges, edgeBoundMask, boundVertices, EH, EF = compute_edge_list(vertices, faces)

    L, vorAreas, _,_ = compute_laplacian(vertices, faces, edges, edgeBoundMask, EF)
    faceNormals, faceAreas = compute_areas_normals(vertices, faces)
    MCNormals, MC, vertexNormals = compute_mean_curvature_normal(vertices, faces, faceNormals, L, vorAreas)

    angleDefect = compute_angle_defect(vertices, faces, boundVertices)
    print("sum angleDefect: ", np.sum(angleDefect))
    GC = angleDefect / vorAreas

    # add a scalar function on the grid
    ps_mesh.add_scalar_quantity("Gaussian Curvature", GC)
    ps_mesh.add_scalar_quantity("Gaussian Curvature Regions", np.sign(GC))
    ps_mesh.add_scalar_quantity("Voronoi Areas", vorAreas)
    ps_mesh.add_scalar_quantity("Signed Mean Curvature", MC)
    ps_mesh.add_vector_quantity("Mean Curvature Normals", MCNormals)
    ps_mesh.add_vector_quantity("Vertex Normals", vertexNormals)

    ps.show()
