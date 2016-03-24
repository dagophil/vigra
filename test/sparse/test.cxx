/************************************************************************/
/*                                                                      */
/*        Copyright 2015-2016 by Ullrich Koethe and Philip Schill       */
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
/*    The above copyright notice and this permission notice shall be    */
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
#include <vigra/sparse/matrix.hxx>
#include <vigra/unittest.hxx>

#include <ostream>

using namespace vigra;

template <typename T0, typename T1, typename T2>
std::ostream & operator<<(std::ostream & s, std::tuple<T0, T1, T2> const & t)
{
    s << std::get<0>(t) << ", " << std::get<1>(t) << ", " << std::get<2>(t);
    return s;
}

struct SparseMatrixTests
{
    /// \brief Test the functional behavior of the sparse matrix type T.
    template <typename T>
    void general_matrix_tests()
    {
        typedef T Matrix;
        typedef typename Matrix::value_type value_type;

        value_type vals[] = {
            10, 0, 0, 12, 0,
            0, 0, 11, 0, 13,
            0, 16, 0, 0, 0,
            0, 0, 11, 0, 13
        };

        // Test nnz, getters, and setters.
        {
            // Test nnz.
            Matrix m(Shape2(4, 5), vals);
            should(m.nnz() == 7);

            // Test the getter.
            should(m.get(0, 0) == 10);
            should(m.get(0, 1) == 0);
            should(m.get(0, 2) == 0);
            should(m.get(0, 3) == 12);
            should(m.get(0, 4) == 0);
            should(m.get(1, 0) == 0);
            should(m.get(1, 1) == 0);
            should(m.get(1, 2) == 11);
            should(m.get(1, 3) == 0);
            should(m.get(1, 4) == 13);
            should(m.get(2, 0) == 0);
            should(m.get(2, 1) == 16);
            should(m.get(2, 2) == 0);
            should(m.get(2, 3) == 0);
            should(m.get(2, 4) == 0);
            should(m.get(3, 0) == 0);
            should(m.get(3, 1) == 0);
            should(m.get(3, 2) == 11);
            should(m.get(3, 3) == 0);
            should(m.get(3, 4) == 13);

            // Test for side effects of the getter.
            should(m.nnz() == 7);

            // Test the setter.
            m.set(1, 4, 25);
            should(m.get(1, 4) == 25);
            should(m.nnz() == 7);
            m.set(3, 0, 35);
            should(m.get(3, 0) == 35);
            should(m.nnz() == 8);
            m.set(3, 4, 0);
            should(m.get(3, 4) == 0);
            should(m.nnz() == 7);
        }

        // Test the proxies.
        {
            Matrix m(Shape2(4, 5), vals);

            // Test getting values from the proxies.
            should(m(0, 0) == 10);
            should(m(0, 1) == 0);
            should(m(0, 2) == 0);
            should(m(0, 3) == 12);
            should(m(0, 4) == 0);
            should(m(1, 0) == 0);
            should(m(1, 1) == 0);
            should(m(1, 2) == 11);
            should(m(1, 3) == 0);
            should(m(1, 4) == 13);
            should(m(2, 0) == 0);
            should(m(2, 1) == 16);
            should(m(2, 2) == 0);
            should(m(2, 3) == 0);
            should(m(2, 4) == 0);
            should(m(3, 0) == 0);
            should(m(3, 1) == 0);
            should(m(3, 2) == 11);
            should(m(3, 3) == 0);
            should(m(3, 4) == 13);

            // Test for side effects of getting values via proxy.
            should(m.nnz() == 7);

            // Test setting values via proxy.
            m(1, 4) = 25;
            should(m.get(1, 4) == 25);
            should(m.nnz() == 7);
            m(3, 0) = 35;
            should(m.get(3, 0) == 35);
            should(m.nnz() == 8);
            m(3, 4) = 0;
            should(m.get(3, 4) == 0);
            should(m.nnz() == 7);

            // Test setting values via proxy using another proxy.
            m(3, 0) = m(1, 4);
            should(m.get(3, 0) == 25);
            should(m.nnz() == 7);
        }
    }

    void test_dok_matrix()
    {
        int vals[] = {
            10, 0, 0, 12, 0,
            0, 0, 11, 0, 13,
            0, 16, 0, 0, 0,
            0, 0, 11, 0, 13
        };
        {
            // Test if the internal map is constructed correctly.
            sparse::DOKMatrix<int> m(Shape2(4, 5), vals);
            auto const & map = m.data();
            should(map.at({0, 0}) == 10);
            should(map.at({0, 3}) == 12);
            should(map.at({1, 2}) == 11);
            should(map.at({1, 4}) == 13);
            should(map.at({2, 1}) == 16);
            should(map.at({3, 2}) == 11);
            should(map.at({3, 4}) == 13);
            should(map.size() == 7);
        }
    }

    void test_coo_matrix()
    {
        int vals[] = {
            10, 0, 0, 12, 0,
            0, 0, 11, 0, 13,
            0, 16, 0, 0, 0,
            0, 0, 11, 0, 13
        };
        {
            // Test if the internal vector is constructed correctly.
            sparse::COOMatrix<int> m(Shape2(4, 5), vals);
            std::vector<std::tuple<size_t, size_t, int> > expected;
            expected.emplace_back(0, 0, 10);
            expected.emplace_back(0, 3, 12);
            expected.emplace_back(1, 2, 11);
            expected.emplace_back(1, 4, 13);
            expected.emplace_back(2, 1, 16);
            expected.emplace_back(3, 2, 11);
            expected.emplace_back(3, 4, 13);
            should(m.data() == expected);
        }
    }
};

struct SparseMatrixTestSuite : public test_suite
{
    SparseMatrixTestSuite()
        :
        test_suite("Sparse matrix test")
    {
        add(testCase(&SparseMatrixTests::general_matrix_tests<sparse::DOKMatrix<int> >));
        add(testCase(&SparseMatrixTests::test_dok_matrix));
        add(testCase(&SparseMatrixTests::general_matrix_tests<sparse::COOMatrix<int> >));
        add(testCase(&SparseMatrixTests::test_coo_matrix));
    }
};

int main(int argc, char** argv)
{
    SparseMatrixTestSuite matrix_test;
    int failed = matrix_test.run(testsToBeExecuted(argc, argv));
    std::cout << matrix_test.report() << std::endl;
    return (failed != 0);
}
