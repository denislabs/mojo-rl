// `MakePolygons`' EMISSION ORDER — libc++'s `std::unordered_map` iteration.
//
// `mjCMesh::MakePolygons` (user_mesh.cc:2904) groups coplanar hull faces in
//
//     std::unordered_map<std::pair<double,double>, MeshPolygon, PairHash>
//
// and then emits `for (const auto& pair : mesh_polygons)`. That iteration
// order is observable: where `multicontact()` breaks a tie between two
// coplanar candidate faces for one edge it takes the FIRST in polygon order,
// so the choice lands in the contact POSITION.
//
// ⚠⚠ IT IS NOT AN IMPLEMENTATION DETAIL WE CAN INVENT AROUND, AND IT IS NOT
// UNREPRODUCIBLE EITHER — that was MEASURED, not assumed. An instrumented
// build of MuJoCo 3.10.0 dumping the real key order shows it is exactly
// bucket-contiguous at the map's own `bucket_count`:
//
//     mesh                  entries   bc     runs  buckets  ratio
//     link_head_tilt_1       1633    3203     862    862    1.000
//     link_d405              1354    1597     811    811    1.000
//     base_link_collision     649     797     419    419    1.000
//
// ratio 1.000 on every mesh — i.e. it IS a libc++ iteration order. And it is
// STABLE across builds: a locally built 3.10.0 and the pixi wheel agree on the
// polygon order of 84 of 85 stretch_3 meshes.
//
// ⚠ SO THE FAITHFUL PORT IS A CALL, exactly as it is for the hull itself
// (`mrl_qhull.c`): MuJoCo does not define this order, its standard library
// does. Reimplementing libc++'s bucket policy — prime growth chain,
// front-insertion, rehash relinking — would be reproducing a specific C++
// standard library from memory inside a physics engine.
//
// ⚠ THE KEYS MUST COME FROM THE RAW FILE VERTICES. `MakePolygons` runs at
// user_mesh.cc:1421, BEFORE `ApplyTransformations` (:1444) and before
// `Process()` bakes the principal frame. Keys computed from `mjModel.mesh_vert`
// are a ROTATION away and overlap the real ones by ~6 of ~1600.

#include <cstddef>
#include <unordered_map>
#include <utility>

namespace {
// Verbatim from user_mesh.cc:2893.
struct PairHash {
  template <class T1, class T2>
  std::size_t operator()(const std::pair<T1, T2>& pair) const {
    return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
  }
};
}  // namespace

extern "C" {

// `keys` is `n` (rtheta, rphi) pairs in FIRST-SEEN order — the order
// `MakePolygons` inserts them in, i.e. the order the hull faces are walked.
// Writes into `out` the insertion indices in the map's ITERATION order.
// Returns the number written, or -1 on a bad argument.
//
// Duplicate keys are ignored after the first, matching MuJoCo's
// `find`-then-`emplace`: only a first sighting creates a node, and only nodes
// are iterated.
int mrl_poly_order(const double* keys, int n, int* out) {
  if (!keys || !out || n < 0) return -1;
  std::unordered_map<std::pair<double, double>, int, PairHash> mp;
  for (int i = 0; i < n; i++) {
    std::pair<double, double> k(keys[2 * i], keys[2 * i + 1]);
    if (mp.find(k) == mp.end()) mp.emplace(k, i);
  }
  int w = 0;
  for (const auto& p : mp) out[w++] = p.second;
  return w;
}

}  // extern "C"
