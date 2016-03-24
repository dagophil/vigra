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
#ifndef VIGRA_SPARSE_MATRIX_HXX
#define VIGRA_SPARSE_MATRIX_HXX

#include <map>
#include <tuple>
#include <vector>
#include <stdexcept>

#include "../multi_shape.hxx"

namespace vigra
{
namespace sparse
{

/// The vigra::sparse namespace contains some sparse matrix formats.
/// See https://en.wikipedia.org/wiki/Sparse_matrix for a description of
/// the formats.
/// There is no range-checking performed. This means that on some matrix
/// types you can create a matrix of shape (2, 3) and call assign a value
/// to the element at (5, 6), which is clearly out-of-range.

namespace detail
{
    // /// \brief The ConstMatrixProxy is used to get elements from a const sparse matrix.
    // template <typename MATRIX>
    // class ConstMatrixProxy
    // {
    // public:

    //     typedef MATRIX Matrix;
    //     typedef typename Matrix::value_type value_type;

    //     ConstMatrixProxy(
    //             Matrix const & matrix, 
    //             size_t i, 
    //             size_t j
    //     )   :
    //         mat_(matrix),
    //         i_(i),
    //         j_(j)
    //     {}

    //     /// \brief If the proxy is converte
    //     operator value_type() const
    //     {
    //         return mat_.get(i_, j_);
    //     }

    //     /// \brief Explicitly delete the assignemnt for the const proxy.
    //     ConstMatrixProxy & operator=(value_type const & v) = delete;

    //     /// \brief Explicitly delete the assignemnt for the const proxy.
    //     ConstMatrixProxy & operator=(ConstMatrixProxy const & other) = delete;

    // private:

    //     Matrix const & mat_; // the referred matrix
    //     size_t const i_; // the first matrix index
    //     size_t const j_; // the second matrix index
    // };

    /// \brief The MatrixProxy is used to get and set elements of a sparse matrix.
    template <typename MATRIX>
    class MatrixProxy
    {
    public:

        typedef MATRIX Matrix;
        typedef typename Matrix::value_type value_type;

        MatrixProxy(
                Matrix & matrix, 
                size_t i, 
                size_t j
        )   :
            mat_(matrix),
            i_(i),
            j_(j)
        {}

        /// \brief Return the matrix element to which the proxy points to.
        operator value_type() const
        {
            return mat_.get(i_, j_);
        }

        /// \brief Set the matrix element to which the proxy points to.
        MatrixProxy & operator=(value_type const & v)
        {
            mat_.set(i_, j_, v);
            return *this;
        }

        /// \brief Set the matrix element to which the proxy points to.
        MatrixProxy & operator=(MatrixProxy const & other)
        {
            return operator=(static_cast<value_type>(other));
        }

    private:

        Matrix & mat_; // the referred matrix
        size_t const i_; // first matrix index
        size_t const j_; // second matrix index
    };

}

/// \brief Dictionary Of Keys matrix.
/// The DOKMatrix consists of a map that maps (row, column)-pairs to the
/// value of the element. Elements that are not present in the map have
/// the value zero.
template <typename T>
class DOKMatrix
{
public:

    typedef T value_type;
    typedef std::pair<size_t, size_t> IndexPair;
    typedef std::map<IndexPair, value_type> Map;
    typedef detail::MatrixProxy<DOKMatrix<T> > Proxy;

    /// \brief Construct a matrix with the given shape from the given values.
    DOKMatrix(
            Shape2 const & shape = Shape2(0, 0),
            value_type* vals = nullptr
    )   :
        shape_(shape)
    {
        if (vals != nullptr)
        {
            for (Int64 i = 0; i < shape[0]; ++i)
            {
                for (Int64 j = 0; j < shape[1]; ++j)
                {
                    set(i, j, vals[i*shape[1]+j]);
                }
            }
        }
    }

    /// \brief Return the matrix shape.
    Shape2 const & shape() const
    {
        return shape_;
    }

    /// \brief Return the number of non-zero elements.
    size_t nnz() const
    {
        return map_.size();
    }

    /// \brief Return the value of the element at (i, j).
    value_type get(
            size_t i,
            size_t j
    ) const {
        auto it = map_.find({i, j});
        if (it != map_.end())
            return it->second;
        else
            return value_type(0);
    }

    /// \brief Set the element at (i, j).
    void set(
            size_t i,
            size_t j,
            value_type const & v
    ){
        if (v == 0)
        {
            // If an element is set to zero, it must be removed from the map.
            auto it = map_.find({i, j});
            if (it != map_.end())
                map_.erase(it);
        }
        else
        {
            // Set the nonzero value.
            map_[{i, j}] = v;
        }
    }

    /// \brief Return a proxy element that can be used to set the element at (i, j).
    Proxy operator()(
            size_t i, 
            size_t j
    ){
        return Proxy(*this, i, j);
    }

    /// \brief Return the value of the element at (i, j).
    /// \note
    /// There is no need for a proxy since this is the const version of operator()
    /// and therefore assigment is not necessary.
    value_type operator()(
            size_t i,
            size_t j
    ) const {
        return get(i, j);
    }

    /// \brief Return the internal map storage.
    Map const & data() const
    {
        return map_;
    }

private:

    Shape2 shape_; // the matrix shape
    Map map_; // the map
};

/// \brief Coordinate list matrix.
/// The COOMatrix stores a vector of (row, column, value) tuples.
/// Elements that are not present in the map have the value zero.
template <typename T>
class COOMatrix
{
public:

    typedef T value_type;
    typedef std::tuple<size_t, size_t, value_type> Triple;
    typedef detail::MatrixProxy<COOMatrix<T> > Proxy;

    COOMatrix(
        Shape2 const & shape = Shape2(0, 0),
        value_type* vals = nullptr
    )   :
        shape_(shape)
    {
        for (size_t i = 0; i < shape_[0]; ++i)
        {
            for (size_t j = 0; j < shape_[1]; ++j)
            {
                auto v = vals[i*shape_[1]+j];
                if (v != 0)
                {
                    data_.emplace_back(i, j, v);
                }
            }
        }
    }

    /// \brief Append a new triple (i, j, v) to the vector with the stored elements.
    /// If appending would destroy the order of the elements (sorted by row index, then column index),
    /// an exception of the type std::runtime_error is thrown.
    void push(size_t i, size_t j, value_type const & v)
    {
        if (!data_.empty())
        {
            auto const li = std::get<0>(data_.back());
            auto const lj = std::get<1>(data_.back());
            if (li > i)
                throw std::runtime_error("COOMatrix::push(): Appending the value would destroy the order of the stored elements (first row index, then column index).");
            else if (li == i && lj > j)
                throw std::runtime_error("COOMatrix::push(): Appending the value would destroy the order of the stored elements (first row index, then column index).");
            else if (li == i && lj == j)
                throw std::runtime_error("COOMatrix::push(): Element exists already.");
        }
        if (v != 0)
            data_.emplace_back(i, j, v);
    }

    /// \brief Return the internal vector storage.
    std::vector<Triple> const & data() const
    {
        return data_;
    }

private:

    Shape2 shape_; // the matrix shape
    std::vector<Triple> data_; // the stored entries, sorted by row index, then column index

};


} // namespace sparse
} // namespace vigra

#endif
