/* mrl_qhull.c — the convex-hull FACE LIST, in qhull's order.
 *
 * WHY THIS EXISTS. `mjCMesh::MakeGraph` (user_mesh.cc) does not implement a
 * convex hull: it calls qhull and then walks `FORALLfacets`. The ORDER of that
 * walk is what `MakePolygons` inserts faces in, which is what
 * `MeshPolygon::Paths()` starts each vertex cycle at, which is what
 * `MakePolygonNormals` reads its first three vertices from. Our own exact hull
 * reproduces qhull's vertex SET but not its facet order, and the resulting
 * polygon normals land up to 0.26 deg from `mesh_polynormal` — against the
 * 0.09167 deg tolerance `alignedFaces` / `alignedFaceEdge` test with. That is
 * the whole of two Menagerie board rows.
 *
 * So this is a TRANSCRIPTION of MakeGraph's extraction loop, not a new
 * algorithm. Keep it that way: every line below has a counterpart there.
 *
 * ⚠ THE INPUT IS THE RAW, DEDUPED VERTEX ARRAY, NOT THE PRINCIPAL-FRAME ONE.
 * `Process()` builds `dvert` from `vert_` BEFORE the principal frame is baked.
 * Measured: hulling the raw vertices reproduces MuJoCo's polygon paths
 * 2580/2580 including the cycle START; hulling the principal-frame vertices
 * reproduces 713/2580, because the float32 round trip perturbs `Qt`'s
 * coplanar tie-breaking.
 */
#include <stdio.h>
#include <stdlib.h>
#include <setjmp.h>
#include <libqhull_r/qhull_ra.h>

/* Fill `faces` with 3*nface global point ids in qhull's facet order.
 *
 *   verts     nvert*3 doubles, the raw deduped vertex array
 *   maxface   capacity of `faces`, in FACES (so 3*maxface ints)
 *   faces     out, 3 ints per face
 *   returns   the face count, or a negative error code:
 *             -1 qhull longjmp'd, -2 not enough room, -3 a non-triangle,
 *             -4 a point id out of range
 */
int mrl_qhull_faces(const double* verts, int nvert, int maxhullvert,
                    int maxface, int* faces) {
  qhT qh_qh;
  qhT* qh = &qh_qh;
  int curlong, totlong, exitcode;
  facetT* facet;
  vertexT *vertex1, **vertex1p;
  int nface = 0, bad = 0;

  /* MakeGraph's option string, including the maxhullvert form. */
  char opt[64];
  if (maxhullvert > -1) {
    snprintf(opt, sizeof opt, "qhull Qt Q9 TA%d", maxhullvert - 4);
  } else {
    snprintf(opt, sizeof opt, "qhull Qt");
  }

  qh_zero(qh, stderr);
  qh_init_A(qh, stdin, stdout, stderr, 0, NULL);
  exitcode = setjmp(qh->errexit);
  qh->NOerrexit = False;
  if (!exitcode) {
    qh_initflags(qh, opt);
    qh_init_B(qh, (double*)verts, nvert, 3, qh_False);
    qh_qhull(qh);
    qh_triangulate(qh);
    qh_vertexneighbors(qh);

    if (qh->num_facets > maxface) {
      bad = -2;
    } else {
      int adr = 0;
      FORALLfacets {
        int ii = 0;
        int ind[3];
        ind[0] = 0; ind[1] = 1; ind[2] = 2;
        /* MakeGraph: reorient faces if flipped. */
        if (facet->toporient) { ind[0] = 1; ind[1] = 0; }
        FOREACHsetelement_(vertexT, facet->vertices, vertex1) {
          int pid;
          if (ii >= 3) { bad = -3; break; }
          pid = qh_pointid(qh, vertex1->point);
          if (pid < 0 || pid >= nvert) { bad = -4; break; }
          faces[adr + ind[ii++]] = pid;
        }
        if (bad) break;
        if (ii != 3) { bad = -3; break; }
        adr += 3;
        nface++;
      }
    }
  } else {
    bad = -1;
  }
  qh_freeqhull(qh, !qh_ALL);
  qh_memfreeshort(qh, &curlong, &totlong);
  return bad ? bad : nface;
}

/* Upper bound on the facet count, so a caller can size `faces` in one pass.
 * Euler for a simplicial 3-polytope: F <= 2V - 4. `Qt` triangulates, so this
 * holds after `qh_triangulate` too. */
int mrl_qhull_maxfaces(int nvert) {
  return nvert < 4 ? 0 : 2 * nvert - 4;
}
