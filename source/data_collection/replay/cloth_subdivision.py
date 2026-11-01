"""Fixed-topology Loop subdivision for recorded cloth; no simulator dependencies."""
import numpy as np
from scipy.sparse import csr_matrix, eye


class LoopSubdivision:
    def __init__(self, num_vertices, faces, levels=2):
        if levels not in (0, 1, 2, 3):
            raise ValueError('Subdivision levels must be 0..3')
        self.levels = levels
        self.num_vertices = num_vertices
        faces = np.asarray(faces, dtype=np.int64)
        if faces.ndim != 2 or faces.shape[1] != 3 or not faces.size:
            raise ValueError('Expected nonempty triangle topology')
        if faces.min() < 0 or faces.max() >= num_vertices:
            raise ValueError('Triangle indices outside recorded vertex range')
        weights = eye(num_vertices, format='csr')
        for _ in range(levels):
            step, faces = self._step(weights.shape[0], faces)
            weights = step @ weights
        self.faces = faces
        self.weights = weights.tocsr()

    @staticmethod
    def _step(n, faces):
        edges = {}
        neighbors = [set() for _ in range(n)]
        for a, b, c in faces:
            if len({a, b, c}) != 3:
                raise ValueError('Degenerate triangle')
            for u, v, opposite in ((a, b, c), (b, c, a), (c, a, b)):
                edges.setdefault(tuple(sorted((u, v))), []).append(opposite)
                neighbors[u].add(v)
                neighbors[v].add(u)
        boundary = [set() for _ in range(n)]
        for (u, v), opposite in edges.items():
            if len(opposite) > 2:
                raise ValueError('Nonmanifold edge; subdivision requires manifold topology')
            if len(opposite) == 1:
                boundary[u].add(v)
                boundary[v].add(u)
        rows, cols, values = [], [], []
        def add(row, col, value):
            rows.append(row); cols.append(col); values.append(value)
        for v in range(n):
            if boundary[v]:
                if len(boundary[v]) != 2:
                    raise ValueError('Nonmanifold boundary vertex')
                add(v, v, .75)
                for j in boundary[v]:
                    add(v, j, .125)
            elif neighbors[v]:
                k = len(neighbors[v])
                beta = (.625 - (.375 + .25 * np.cos(2 * np.pi / k)) ** 2) / k
                add(v, v, 1 - k * beta)
                for j in neighbors[v]:
                    add(v, j, beta)
            else:
                add(v, v, 1.)
        edge_ids = {}
        for edge in sorted(edges):
            row = n + len(edge_ids)
            edge_ids[edge] = row
            opposite = edges[edge]
            for j in edge:
                add(row, j, .5 if len(opposite) == 1 else .375)
            if len(opposite) == 2:
                for j in opposite:
                    add(row, j, .125)
        refined = []
        for a, b, c in faces:
            ab, bc, ca = (edge_ids[tuple(sorted(e))] for e in ((a, b), (b, c), (c, a)))
            refined.extend(((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)))
        return csr_matrix((values, (rows, cols)), shape=(n + len(edges), n)), np.asarray(refined)

    def apply(self, points):
        points = np.asarray(points)
        if points.shape != (self.num_vertices, 3) or not np.isfinite(points).all():
            raise ValueError(f'Expected finite ({self.num_vertices}, 3) trajectory points')
        return np.asarray(self.weights @ points, dtype=np.float32)

    def face_varying(self, values):
        """Linearly refine corner attributes independently to preserve UV seams."""
        values = np.asarray(values)
        for _ in range(self.levels):
            a, b, c = values[:, 0], values[:, 1], values[:, 2]
            ab, bc, ca = (a + b) * .5, (b + c) * .5, (c + a) * .5
            values = np.stack([np.stack(x, axis=1) for x in
                               ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))], axis=1)
            values = values.reshape((-1, 3) + values.shape[3:])
        return values


class SubdividedUsdMesh:
    """Author refinement in the replay stage's edit layer, keeping material bindings."""
    def __init__(self, mesh, levels):
        from pxr import UsdGeom, Vt
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
        if not np.all(counts == 3):
            raise ValueError('Cloth subdivision requires triangular faces')
        points = np.asarray(mesh.GetPointsAttr().Get())
        self.mapping = LoopSubdivision(len(points), np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3), levels)
        attrs = []
        for pv in UsdGeom.PrimvarsAPI(mesh).GetPrimvars():
            value = pv.ComputeFlattened()
            if value is None or pv.GetInterpolation() == 'constant':
                continue
            arr = np.asarray(value)
            mode = pv.GetInterpolation()
            if mode == 'faceVarying':
                result = self.mapping.face_varying(arr.reshape((len(counts), 3) + arr.shape[1:]))
                result = result.reshape((-1,) + arr.shape[1:])
            elif mode == 'uniform':
                result = np.repeat(arr, 4 ** levels, axis=0)
            elif mode in ('vertex', 'varying') and np.issubdtype(arr.dtype, np.floating):
                result = (self.mapping.weights @ arr.reshape(len(points), -1)).reshape((-1,) + arr.shape[1:])
            else:
                raise ValueError(f'Unsupported cloth attribute {pv.GetName()} ({mode})')
            attrs.append((pv, type(value)(result.tolist())))
        if UsdGeom.Subset.GetAllGeomSubsets(mesh):
            raise ValueError('Cloth material subsets need explicit refinement')
        for pv, values in attrs:
            pv.Set(values)
            if pv.IsIndexed():
                pv.BlockIndices()
        self.mesh = mesh
        mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * len(self.mapping.faces)))
        mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(self.mapping.faces.ravel().tolist()))
        mesh.GetSubdivisionSchemeAttr().Set('none')
        mesh.SetNormalsInterpolation('vertex')
        self.update(points)

    def update(self, points):
        from pxr import Vt
        points = self.mapping.apply(points)
        faces = self.mapping.faces
        tri = points[faces]
        normals = np.zeros_like(points)
        fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        for corner in range(3):
            np.add.at(normals, faces[:, corner], fn)
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-20)
        self.mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))
        self.mesh.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(normals))
        self.mesh.GetExtentAttr().Set(Vt.Vec3fArray.FromNumpy(np.array([points.min(0), points.max(0)])))
