# Detect HDF5 installation via Rhdf5lib
AC_INIT([scLean], [0.1.0])

# Check for R
if test -z "${R_HOME}"; then
    AC_MSG_ERROR([R_HOME is not set])
fi

# Use Rhdf5lib to find HDF5 flags
PKG_CXX_HDF5_CXXFLAGS=`${R_HOME}/bin/Rscript -e "cat(Rhdf5lib::pkgconfig('PKG_CXXFLAGS'))" 2>/dev/null`
PKG_CXX_HDF5_LIBS=`${R_HOME}/bin/Rscript -e "cat(Rhdf5lib::pkgconfig('PKG_LIBS'))" 2>/dev/null`

# Check OpenMP support
AC_OPENMP
if test -n "${OPENMP_CFLAGS}"; then
    AC_SUBST([OPENMP_CXXFLAGS], ['$(SHLIB_OPENMP_CXXFLAGS)'])
    AC_SUBST([OPENMP_LIBS], ['$(SHLIB_OPENMP_CXXFLAGS)'])
else
    AC_SUBST([OPENMP_CXXFLAGS], [""])
    AC_SUBST([OPENMP_LIBS], [""])
fi

AC_SUBST([HDF5_CXXFLAGS], ["${PKG_CXX_HDF5_CXXFLAGS}"])
AC_SUBST([HDF5_LIBS], ["${PKG_CXX_HDF5_LIBS}"])

# Check C++17 support
AX_CXX_COMPILE_STDCXX(17, noext, mandatory)

AC_OUTPUT([src/Makevars])
