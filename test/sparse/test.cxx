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
    void test_dok_matrix()
    {
        int vals[] = {
            10, 0, 0, 12, 0,
            0, 0, 11, 0, 13,
            0, 16, 0, 0, 0,
            0, 0, 11, 0, 13
        };
        {
            sparse::DOKMatrix<int> m(Shape2(4, 5), vals);
            should(m.nnz() == 7);

            // Make sure that the map only contains the nonzero values.
            auto const & map = m.data();
            should(map.at({0, 0}) == 10);
            should(map.at({0, 3}) == 12);
            should(map.at({1, 2}) == 11);
            should(map.at({1, 4}) == 13);
            should(map.at({2, 1}) == 16);
            should(map.at({3, 2}) == 11);
            should(map.at({3, 4}) == 13);
            should(map.size() == 7);

            // Test the getter.
            should(m.get(0, 0) == 10);
            should(m.get(2, 1) == 16);
            should(m.get(3, 4) == 13);

            // Test the setter.
            m.set(1, 4, 25);
            should(m.get(1, 4) == 25);
            should(m.data().size() == 7);
        }
        {
            sparse::DOKMatrix<int> m(Shape2(4, 5), vals);
            should(m.nnz() == 7);

            // Test getting values from the proxy.
            should(m.get(1, 4) == m(1, 4));
            should(m.get(0, 1) == m(0, 1));
            should(m.nnz() == 7); // test for side effects

            // Test setting values from the proxy.
            m(1, 4) = 15;
            should(m.get(1, 4) == 15);
            m(1, 4) = 0;
            should(m.get(1, 4) == 0);
            should(m.nnz() == 6); // make sure that the map size was actually decreased

            // Test setting values from a second proxy.
            m(1, 4) = m(0, 0);
            should(m.get(1, 4) == 10);
            should(m.nnz() == 7); // make sure that the map size was actually increased
        }
        {
            sparse::DOKMatrix<int> const m0(Shape2(4, 5), vals);
            should(m0(1, 4) == 13);

            // The following lines should not compile, since we cannot set the elements of a const matrix.
            // FIXME: How can you test this?
            // m0(1, 4) = m0(2, 4);
            // m0(1, 4) = 5;
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
            sparse::COOMatrix<int> m(Shape2(4, 5), vals);

            // Make sure that the internal vector is constructed correctly.
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
        add(testCase(&SparseMatrixTests::test_dok_matrix));
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
