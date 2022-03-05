// main.h adds instrumentation which breaks constexpr so we do not run any tests in here,
// this is strictly compile-time.
#include <Eigen/Dense>
#include <Eigen/Geometry>

using namespace Eigen;

inline void error_if_not_constexpr() {} // not constexpr
#if EIGEN_COMP_CXXVER >= 20
consteval
#else
constexpr
#endif
void assert_constexpr(bool b) {
  if (!b) error_if_not_constexpr();
}

constexpr bool zeroSized()
{
  constexpr Matrix<float, 0, 0> m0;
  static_assert(m0.size() == 0, "");

  constexpr Matrix<float, 0, 0> m1;
  static_assert(m0 == m1, "");
  static_assert(!(m0 != m1), "");

  constexpr Array<int, 0, 0> a0;
  static_assert(a0.size() == 0, "");

  constexpr Array<int, 0, 0> a1;
  static_assert((a0 == a1).all(), "");
  static_assert((a0 != a1).count() == 0, "");

  constexpr Array<float, 0, 0> af;
  static_assert(m0 == af.matrix(), "");
  static_assert((m0.array() == af).all(), "");
  static_assert(m0.array().matrix() == m0, "");

  return true;
}

static_assert(zeroSized(), "");

static constexpr double static_data[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };

constexpr bool maps()
{
  constexpr Map<const Vector4d> m(static_data);
  static_assert(m(0) == 1, "");
  constexpr Map<const Array<double, 4, 1>> a(static_data);
  static_assert(m == a.matrix(), "");
  static_assert(m.size() == 4, "");
  static_assert(a.size() == 4, "");
  static_assert(m.rows() == 4 && m.cols() == 1, "");
  return true;
}

static_assert(maps(), "");

constexpr bool nc_maps()
{
  bool result = true;

  double d[] = {1, 2, 3, 4};
  Map<Vector4d> m(d);
  result = result && (m.x() == 1 && m.y() == 2 && m.z() == 3 && m.w() == 4);

  float array[3] = {};
  auto v = Vector3f::Map(array);
  v.fill(10);
  result = result && (v.array() == 10).all();

  return result;
}

constexpr bool blocks()
{
  constexpr Map<const Matrix2d> m(static_data);
  constexpr auto block = m.block<2,1>(0, 1);

  constexpr Map<const Vector2d> v(static_data + 2);
  static_assert(block == v, "");

  return true;
}

static_assert(blocks(), "");

constexpr bool diagonal_row_columns()
{
  constexpr Map<const Matrix2d> m(static_data);
  static_assert(m.block<2,1>(0, 1) == m.col(1), "");
  static_assert(m.block<1,2>(1, 0) == m.row(1), "");
  static_assert(m.diagonal()(0) == 1 && m.diagonal()(1) == 4, "");
  return true;
}

static_assert(diagonal_row_columns(), "");

static constexpr int static_data_antisym[] = {
  0, 1, -1,
  -1, 0, 1,
  1, -1, 0 };

constexpr bool transpose_unaryminus()
{
  constexpr Map<const Matrix<int, 3, 3>> m(static_data_antisym);

  static_assert(m.transpose() == -m, "");
  static_assert(-m.transpose() == m, "");
  static_assert((-m).transpose() == m, "");

  static_assert(m.transpose() != m, "");
  static_assert(-m.transpose() != -m, "");
  static_assert((-m).transpose() != -m, "");

  return true;
}

static_assert(transpose_unaryminus(), "");

constexpr bool reductions()
{
  constexpr Map<const Matrix<int, 3, 3>> m(static_data_antisym);
  static_assert(m.size() == 9, "");

  return true;
}

static_assert(reductions(), "");

constexpr bool scalar_mult_div()
{
  constexpr Map<const Matrix2d> m(static_data);

  static_assert((m * 2)(0,0) == 2, "");
  static_assert((m / 2)(1,1) == 2*m(0,0), "");

  constexpr double c = 8;
  static_assert((m * c)(0,0) == 8, "");
  static_assert((m.array() / c).matrix() == 1/c * m, "");
  return true;
}

static_assert(scalar_mult_div(), "");

constexpr bool constant_identity()
{
  static_assert(Matrix3f::Zero()(0,0) == 0, "");
  static_assert(Matrix4d::Ones()(3,3) == 1, "");
  static_assert(Matrix2i::Identity()(0,0) == 1 && Matrix2i::Identity()(1,0) == 0, "");
  static_assert(Matrix<float, Dynamic, Dynamic>::Ones(2,3).size() == 6, "");
  static_assert(Matrix<float, Dynamic, 1>::Zero(10).rows() == 10, "");

  return true;
}

static_assert(constant_identity(), "");

constexpr bool dynamic_basics()
{
  // This verifies that we only calculate the entry that we need.
  static_assert(Matrix<double, Dynamic, Dynamic>::Identity(50000,50000).array()(25,25) == 1, "");

  static_assert(Matrix4d::Identity().block(1,1,2,2)(0,1) == 0, "");
  static_assert(MatrixXf::Identity(50,50).transpose() == MatrixXf::Identity(50, 50), "");

  constexpr Map<const MatrixXi> dynMap(static_data_antisym, 3, 3);
  constexpr Map<const Matrix3i> staticMap(static_data_antisym);
  static_assert(dynMap == staticMap, "");
  static_assert(dynMap.transpose() != staticMap, "");
  // e.g. this hits an assertion at compile-time that would otherwise fail at runtime.
  //static_assert(dynMap != staticMap.block(1,2,0,0));

  return true;
}

static_assert(dynamic_basics(), "");

constexpr bool sums()
{
  constexpr Map<const Matrix<double, 4, 4>> m(static_data);
  constexpr auto b(m.block<2,2>(0,0)); // 1 2 5 6
  constexpr Map<const Matrix2d> m2(static_data); // 1 2 3 4

  static_assert((b + m2).col(0) == 2*Map<const Vector2d>(static_data), "");
  static_assert(b + m2 == m2 + b, "");

  static_assert((b - m2).col(0) == Vector2d::Zero(), "");
  static_assert((b - m2).col(1) == 2*Vector2d::Ones(), "");

  static_assert((2*b - m2).col(0) == b.col(0), "");
  static_assert((b - 2*m2).col(0) == -b.col(0), "");

  static_assert((b - m2 + b + m2 - 2*b) == Matrix2d::Zero(), "");

  return true;
}

static_assert(sums(), "");

constexpr bool unit_vectors()
{
  static_assert(Vector4d::UnitX()(0) == 1, "");
  static_assert(Vector4d::UnitY()(1) == 1, "");
  static_assert(Vector4d::UnitZ()(2) == 1, "");
  static_assert(Vector4d::UnitW()(3) == 1, "");

  static_assert(Vector4d::UnitX().dot(Vector4d::UnitX()) == 1, "");
  static_assert(Vector4d::UnitX().dot(Vector4d::UnitY()) == 0, "");
  static_assert((Vector4d::UnitX() + Vector4d::UnitZ()).dot(Vector4d::UnitY() + Vector4d::UnitW()) == 0, "");

  return true;
}

static_assert(unit_vectors(), "");

constexpr bool construct_from_other()
{
  return true;
}

static_assert(construct_from_other(), "");

constexpr bool construct_from_values()
{
  return true;
}

static_assert(construct_from_values(), "");

constexpr bool triangular()
{
  bool result = true;

  return result;
}

static_assert(triangular(), "");

constexpr bool nc_construct_from_values()
{
  bool result = true;

  return result;
}

constexpr bool nc_crossproduct()
{
  bool result = true;
  return result;
}

constexpr bool nc_cast()
{
  bool result = true;

  return result;
}

constexpr bool nc_product()
{
  bool result = true;
  return result;
}

static constexpr double static_data_quat[] = { 0, 1, 1, 0 };

constexpr bool nc_quat_mult()
{
  bool result = static_data_quat[3] == 0; // Silence warning about unused with C++14

  return result;
}

// Run tests that aren't explicitly constexpr.
constexpr bool test_nc()
{
  assert_constexpr(nc_maps());
  assert_constexpr(nc_construct_from_values());
  assert_constexpr(nc_crossproduct());
  assert_constexpr(nc_cast());
  assert_constexpr(nc_product());
  assert_constexpr(nc_quat_mult());
  return true;
}

int main()
{
  return !test_nc();
}
