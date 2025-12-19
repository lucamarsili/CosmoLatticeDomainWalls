/* config.h.  Generated from config.h.in by configure.  */
/* config.h.in.  Generated from configure.ac by autoheader.  */

/* Define to disable Fortran wrappers. */
#define DISABLE_FORTRAN 1

/* Define to dummy `main' function (if any) required to link to the Fortran
   libraries. */
/* #undef FC_DUMMY_MAIN */

/* Define if F77 and FC dummy `main' functions are identical. */
/* #undef FC_DUMMY_MAIN_EQ_F77 */

/* Define to a macro mangling the given C identifier (in lower and upper
   case), which must not contain underscores, for linking with Fortran. */
/* #undef FC_FUNC */

/* As FC_FUNC, but for C identifiers containing underscores. */
/* #undef FC_FUNC_ */

/* Define if FC_FUNC and FC_FUNC_ are equivalent. */
/* #undef FC_FUNC_EQUIV */

/* Define to alternate name for `main' routine that is called from a `main' in
   the Fortran libraries. */
/* #undef FC_MAIN */

/* Define to the prefix of the namespace of the FFTW library. */
#define FFTW_PREFIX 

/* Define to 1 if you have the <dlfcn.h> header file. */
#define HAVE_DLFCN_H 1

/* Define to 1 if you have the <inttypes.h> header file. */
#define HAVE_INTTYPES_H 1

/* Define to 1 if you have the `m' library (-lm). */
#define HAVE_LIBM 1

/* Define to 1 if the compiler supports 'long double' */
#define HAVE_LONG_DOUBLE 1

/* Define to 1 if you have the <memory.h> header file. */
#define HAVE_MEMORY_H 1

/* Define if you have the MPI library. */
#define HAVE_MPI 1

/* Define if OpenMP is enabled */
#define HAVE_OPENMP 1

/* Define to 1 if the system has the type `ptrdiff_t'. */
#define HAVE_PTRDIFF_T 1

/* Define to 1 if you have the <stdint.h> header file. */
#define HAVE_STDINT_H 1

/* Define to 1 if you have the <stdlib.h> header file. */
#define HAVE_STDLIB_H 1

/* Define to 1 if you have the <strings.h> header file. */
#define HAVE_STRINGS_H 1

/* Define to 1 if you have the <string.h> header file. */
#define HAVE_STRING_H 1

/* Define to 1 if you have the <sys/stat.h> header file. */
#define HAVE_SYS_STAT_H 1

/* Define to 1 if you have the <sys/types.h> header file. */
#define HAVE_SYS_TYPES_H 1

/* Define to 1 if you have the <unistd.h> header file. */
#define HAVE_UNISTD_H 1

/* Define to the sub-directory in which libtool stores uninstalled libraries.
   */
#define LT_OBJDIR ".libs/"

/* Name of package */
#define PACKAGE "pfft"

/* Define to the address where bug reports for this package should be sent. */
#define PACKAGE_BUGREPORT "michael.pippig@mathematik.tu-chemnitz.de"

/* Define to the full name of this package. */
#define PACKAGE_NAME "PFFT"

/* Define to the full name and version of this package. */
#define PACKAGE_STRING "PFFT 1.0.8-alpha"

/* Define to the one symbol short name of this package. */
#define PACKAGE_TARNAME "pfft"

/* Define to the home page for this package. */
#define PACKAGE_URL "http://www.tu-chemnitz.de/~mpip/software/"

/* Define to the version of this package. */
#define PACKAGE_VERSION "1.0.8-alpha"

/* Define to enable extra debugging code. */
/* #undef PFFT_DEBUG */

/* Define to compile in double precision. */
#define PFFT_PREC_DOUBLE 1

/* Define to compile in long-double precision. */
/* #undef PFFT_PREC_LDOUBLE */

/* Define to compile in single precision. */
/* #undef PFFT_PREC_SINGLE */

/* The size of `int', as computed by sizeof. */
/* #undef SIZEOF_INT */

/* The size of `MPI_Fint', as computed by sizeof. */
#define SIZEOF_MPI_FINT 4

/* The size of `ptrdiff_t', as computed by sizeof. */
#define SIZEOF_PTRDIFF_T 8

/* Define to 1 if you have the ANSI C header files. */
#define STDC_HEADERS 1

/* Version number of package */
#define VERSION "1.0.8-alpha"

/* Include gfortran-compatible wrappers in addition to any other Fortran
   wrappers. */
/* #undef WITH_GFORTRAN_WRAPPERS */

/* Define to `__inline__' or `__inline' if that's what the C compiler
   calls it, or to nothing if 'inline' is not supported under any name.  */
#ifndef __cplusplus
/* #undef inline */
#endif
