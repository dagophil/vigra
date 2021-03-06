/************************************************************************/
/*                                                                      */
/*       Copyright 2011 by Ullrich Koethe and Michael Hanselmann        */
/*                                                                      */
/*    This file is part of the VIGRA computer vision library.           */
/*    The VIGRA Website is                                              */
/*        http://hci.iwr.uni-heidelberg.de/vigra/                       */
/*    Please direct questions, bug reports, and contributions to        */
/*        ullrich.koethe@iwr.uni-heidelberg.de    or                    */
/*        vigra@informatik.uni-hamburg.de                               */
/*                                                                      */
/*    Permission is hereby granted, free of charge, to any person       */
/*    obtaining a copy of this software and associated documentation    */
/*    files (the "Software"), to deal in the Software without           */
/*    restriction, including without limitation the rights to use,      */
/*    copy, modify, merge, publish, distribute, sublicense, and/or      */
/*    sell copies of the Software, and to permit persons to whom the    */
/*    Software is furnished to do so, subject to the following          */
/*    conditions:                                                       */
/*                                                                      */
/*    The above copyrigfht notice and this permission notice shall be    */
/*    included in all copies or substantial portions of the             */
/*    Software.                                                         */
/*                                                                      */
/*    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND    */
/*    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES   */
/*    OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND          */
/*    NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT       */
/*    HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,      */
/*    WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING      */
/*    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR     */
/*    OTHER DEALINGS IN THE SOFTWARE.                                   */
/*                                                                      */
/************************************************************************/

#define PY_ARRAY_UNIQUE_SYMBOL vigranumpylearning_PyArray_API
// #define NO_IMPORT_ARRAY

#include <vigra/numpy_array.hxx>
#include <vigra/numpy_array_converters.hxx>
#include <vigra/unsupervised_decomposition.hxx>
#include <set>
#include <cmath>
#include <memory>
#include <boost/python.hpp>

namespace python = boost::python;
namespace vigra
{

template<class U>
python::tuple
pythonPCA(NumpyArray<2,U> features, int nComponents)
{
    vigra_precondition(!features.axistags(),
                       "principalComponents(): feature matrix must not have axistags\n"
                       "(use 'array.view(numpy.ndarray)' to remove them).");

    NumpyArray<2, U> fz(Shape2(features.shape(0), nComponents));
    NumpyArray<2, U> zv(Shape2(nComponents, features.shape(1)));

    {
        PyAllowThreads _pythread;
        principalComponents(features, fz, zv);
    }
    return python::make_tuple(fz, zv);
}

template<class U>
python::tuple
pythonPLSA(NumpyArray<2,U> features,
           int nComponents,
           int nIterations,
           double minGain,
           bool normalize)
{
    vigra_precondition(!features.axistags(),
                       "pLSA(): feature matrix must not have axistags\n"
                       "(use 'array.view(numpy.ndarray)' to remove them).");

    NumpyArray<2, U> fz(Shape2(features.shape(0), nComponents));
    NumpyArray<2, U> zv(Shape2(nComponents, features.shape(1)));

    {
        PyAllowThreads _pythread;
        pLSA(features, fz, zv,
             RandomNumberGenerator<>(),
             PLSAOptions().maximumNumberOfIterations(nIterations)
                          .minimumRelativeGain(minGain)
                          .normalizedComponentWeights(normalize));
    }
    return python::make_tuple(fz, zv);
}


void defineUnsupervised()
{
    using namespace python;

    docstring_options doc_options(true, true, false);

    def("principalComponents", registerConverters(&pythonPCA<double>),
        (arg("features"), arg("nComponents")),
        "\nPerform principal component analysis. \n\n"
        "The imput matrix 'features' must have shape (nFeatures*nSamples). PCA will\n"
        "reduce it to a smaller matrix 'C' with shape (nComponents*nSamples) that \n"
        "preserves as much variance as possible. Specifically, the call::\n\n"
        "    P, C = principalComponents(features, 3)\n\n"
        "returns a projection matrix 'P' with shape (nComponents*nFeatures)\n"
        "such that ``C = numpy.dot(numpy.transpose(P), features)``. Conversely, the\n"
        "matrix  ``f = numpy.dot(P, C)`` is the best possible rank-nComponents\n"
        "approximation to the matrix 'features' under the least-squares criterion.\n\n"
        "See principalComponents_ in the C++ documentation for more detailed\ninformation.\n\n");

    PLSAOptions options;

    def("pLSA", registerConverters(&pythonPLSA<double>),
        (arg("features"), arg("nComponents"), arg("nIterations") = options.max_iterations,
         arg("minGain") = options.min_rel_gain, arg("normalize") = options.normalized_component_weights),
        "\nPerform probabilistic latent semantic analysis. \n\n"
        "The imput matrix 'features' must have shape (nFeatures*nSamples). PCA will\n"
        "reduce it to a smaller matrix 'C' with shape (nComponents*nSamples) that \n"
        "preserves as much information as possible. Specifically, the call::\n\n"
        "    P, C = pLSA(features, 3)\n\n"
        "returns a projection matrix 'P' with shape (nComponents*nFeatures)\n"
        "such that the matrix ``f = numpy.dot(P, C)`` is a rank-nComponents matrix\n"
        "that approximates the matrix 'features' well under the pLSA criterion.\n"
        "Note that the result of pLSA() is not unique, since the algorithm uses random\n"
        "initialization.\n\n"
        "See pLSA_ in the C++ documentation for more detailed\ninformation.\n\n");
}

void defineRandomForest();
void defineRandomForestOld();

} // namespace vigra


using namespace vigra;
using namespace boost::python;

BOOST_PYTHON_MODULE_INIT(learning)
{
    import_vigranumpy();
    defineUnsupervised();
    defineRandomForest();
    defineRandomForestOld();
}


