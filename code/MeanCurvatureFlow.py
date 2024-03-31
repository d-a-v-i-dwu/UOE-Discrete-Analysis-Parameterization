import os
import polyscope as ps
import polyscope.imgui as psim
import numpy as np
from DAFunctions import load_off_file, compute_areas_normals, compute_laplacian, compute_mean_curvature_normal, compute_edge_list, mean_curvature_flow


def callback():
    # Executed every frame
    global flowRate, isFlowing, currVertices

    # UI stuff
    psim.PushItemWidth(50)

    psim.TextUnformatted("Flow Parameters")
    psim.Separator()
    changed, isFlowing = psim.Checkbox("Flowing", isFlowing)

    psim.PopItemWidth()

    # Actual animation
    if not isFlowing:
        return

    currVertices = mean_curvature_flow(faces, boundVertices, currVertices, L, vorAreas, flowRate, isExplicit)
    # currL, currVorAreas = compute_laplacian(vertices, faces, edges, edgeBoundMask, EF)
    # faceNormals, faceAreas = compute_areas_normals(vertices, faces)
    # MCNormals, MC, vertexNormals = compute_mean_curvature_normal(currVertices, faces, faceNormals, currL, currVorAreas)

    # ps_mesh.add_scalar_quantity("Signed Mean Curvature", MC)
    # ps_mesh.add_vector_quantity("Mean Curvature Normals", MCNormals)
    ps_mesh.update_vertex_positions(currVertices)


if __name__ == '__main__':
    ps.init()

    vertices, faces = load_off_file(os.path.join('data', 'lion-head.off'))

    isExplicit = False
    isFlowing = False

    ps_mesh = ps.register_surface_mesh("Input Mesh", vertices, faces)
    currVertices = vertices
    halfedges, edges, edgeBoundMask, boundVertices, EH, EF = compute_edge_list(vertices, faces)

    #Initial values
    L, vorAreas,_,_ = compute_laplacian(vertices, faces, edges, edgeBoundMask, EF)
    faceNormals, faceAreas = compute_areas_normals(vertices, faces)
    MCNormals, MC, vertexNormals = compute_mean_curvature_normal(vertices, faces, faceNormals, L, vorAreas)
    ps_mesh.add_scalar_quantity("Signed Mean Curvature", MC)
    ps_mesh.add_vector_quantity("Mean Curvature Normals", MCNormals)

    flowRate = 50000 * np.min(vorAreas)

    ps.set_user_callback(callback)
    ps.show()
    ps.clear_user_callback()
