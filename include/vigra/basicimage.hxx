/************************************************************************/
/*                                                                      */
/*               Copyright 1998-2002 by Ullrich Koethe                  */
/*       Cognitive Systems Group, University of Hamburg, Germany        */
/*                                                                      */
/*    This file is part of the VIGRA computer vision library.           */
/*    You may use, modify, and distribute this software according       */
/*    to the terms stated in the LICENSE file included in               */
/*    the VIGRA distribution.                                           */
/*                                                                      */
/*    The VIGRA Website is                                              */
/*        http://kogs-www.informatik.uni-hamburg.de/~koethe/vigra/      */
/*    Please direct questions, bug reports, and contributions to        */
/*        koethe@informatik.uni-hamburg.de                              */
/*                                                                      */
/*  THIS SOFTWARE IS PROVIDED AS IS AND WITHOUT ANY EXPRESS OR          */
/*  IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED      */
/*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. */
/*                                                                      */
/************************************************************************/

#ifndef VIGRA_BASICIMAGE_HXX
#define VIGRA_BASICIMAGE_HXX

#include <memory>
#include <algorithm>
#include "vigra/utilities.hxx"
#include "vigra/iteratortraits.hxx"
#include "vigra/accessor.hxx"

namespace vigra {

template <class IMAGEITERATOR>
class LineBasedColumnIteratorPolicy
{
  public:
    typedef IMAGEITERATOR                             ImageIterator;
    typedef typename IMAGEITERATOR::LineStartIterator LineStartIterator;
    typedef typename IMAGEITERATOR::value_type        value_type;
    typedef typename IMAGEITERATOR::difference_type::MoveY
                                                      difference_type;
    typedef typename IMAGEITERATOR::reference         reference;
    typedef typename IMAGEITERATOR::index_reference   index_reference;
    typedef typename IMAGEITERATOR::pointer           pointer;
    typedef std::random_access_iterator_tag           iterator_category;


    struct BaseType
    {
        explicit BaseType(LineStartIterator c = LineStartIterator(),
                          difference_type o = 0)
        : line_start_(c), offset_(o)
        {}

        LineStartIterator line_start_;
        difference_type offset_;
    };

    static void initialize(BaseType &) {}

    static reference dereference(BaseType const & d)
        { return const_cast<reference>(*(*d.line_start_ + d.offset_)); }

    static index_reference dereference(BaseType const & d, difference_type n)
    {
        return const_cast<index_reference>(*(d.line_start_[n] + d.offset_));
    }

    static bool equal(BaseType const & d1, BaseType const & d2)
        { return d1.line_start_ == d2.line_start_; }

    static bool less(BaseType const & d1, BaseType const & d2)
        { return d1.line_start_ < d2.line_start_; }

    static difference_type difference(BaseType const & d1, BaseType const & d2)
        { return d1.line_start_ - d2.line_start_; }

    static void increment(BaseType & d)
        { ++d.line_start_; }

    static void decrement(BaseType & d)
        { --d.line_start_; }

    static void advance(BaseType & d, difference_type n)
        { d.line_start_ += n; }
};

/********************************************************/
/*                                                      */
/*                    BasicImageIterator                */
/*                                                      */
/********************************************************/

/** Implementation of the standard image iterator for \ref vigra::BasicImage.
    See \ref vigra::ImageIterator for documentation.

    <b>\#include</b> "<a href="basicimage_8hxx-source.html">vigra/basicimage.hxx</a>"
    Namespace: vigra
*/
template <class IMAGEITERATOR, class PIXELTYPE,
          class REFERENCE, class POINTER, class LINESTARTITERATOR>
class BasicImageIteratorBase
{
  public:
    typedef BasicImageIteratorBase<IMAGEITERATOR,
            PIXELTYPE, REFERENCE, POINTER, LINESTARTITERATOR> self_type;

    typedef LINESTARTITERATOR    LineStartIterator;
    typedef PIXELTYPE            value_type;
    typedef PIXELTYPE            PixelType;
    typedef REFERENCE            reference;
    typedef REFERENCE            index_reference;
    typedef POINTER              pointer;
    typedef Diff2D               difference_type;
    typedef image_traverser_tag  iterator_category;
    typedef POINTER              row_iterator;
    typedef IteratorAdaptor<LineBasedColumnIteratorPolicy<IMAGEITERATOR> >
                                 column_iterator;

    typedef int                  MoveX;
    typedef LINESTARTITERATOR    MoveY;

    MoveX x;
    MoveY y;

    IMAGEITERATOR & operator+=(difference_type const & s)
    {
        x += s.x;
        y += s.y;
        return static_cast<IMAGEITERATOR &>(*this);
    }

    IMAGEITERATOR & operator-=(difference_type const & s)
    {
        x -= s.x;
        y -= s.y;
        return static_cast<IMAGEITERATOR &>(*this);
    }

    IMAGEITERATOR operator+(difference_type const & s) const
    {
        IMAGEITERATOR ret(static_cast<IMAGEITERATOR const &>(*this));

        ret += s;

        return ret;
    }

    IMAGEITERATOR operator-(difference_type const & s) const
    {
        IMAGEITERATOR ret(static_cast<IMAGEITERATOR const &>(*this));

        ret -= s;

        return ret;
    }

    difference_type operator-(BasicImageIteratorBase const & rhs) const
    {
        return difference_type(x - rhs.x, y - rhs.y);
    }

    bool operator==(BasicImageIteratorBase const & rhs) const
    {
        return (x == rhs.x) && (y == rhs.y);
    }

    bool operator!=(BasicImageIteratorBase const & rhs) const
    {
        return (x != rhs.x) || (y != rhs.y);
    }

    reference operator*() const
    {
        return *(*y + x );
    }

    pointer operator->() const
    {
        return *y + x;
    }

    index_reference operator[](difference_type const & d) const
    {
        return *(*(y + d.y) + x + d.x);
    }

    index_reference operator()(int dx, int dy) const
    {
        return *(*(y + dy) + x + dx);
    }

    pointer operator[](int dy) const
    {
        return y[dy] + x;
    }

    row_iterator rowIterator() const
        { return *y + x; }

    column_iterator columnIterator() const
    {
        typedef typename column_iterator::BaseType Iter;
        return column_iterator(Iter(y, x));
    }

  protected:
    BasicImageIteratorBase(LINESTARTITERATOR const & line)
    : x(0),
      y(line)
    {}

    BasicImageIteratorBase(int ix, LINESTARTITERATOR const & line)
    : x(ix),
      y(line)
    {}

    BasicImageIteratorBase()
    : x(0),
      y(0)
    {}
};

/********************************************************/
/*                                                      */
/*                    BasicImageIterator                */
/*                                                      */
/********************************************************/

/** Implementation of the standard image iterator for \ref vigra::BasicImage.
    See \ref vigra::ImageIterator for documentation.

    <b>\#include</b> "<a href="basicimage_8hxx-source.html">vigra/basicimage.hxx</a>"
    Namespace: vigra
*/
template <class PIXELTYPE, class ITERATOR>
class BasicImageIterator
: public BasicImageIteratorBase<BasicImageIterator<PIXELTYPE, ITERATOR>,
                            PIXELTYPE, PIXELTYPE &, PIXELTYPE *, ITERATOR>
{
  public:

    typedef BasicImageIteratorBase<BasicImageIterator, PIXELTYPE,
                                PIXELTYPE &, PIXELTYPE *, ITERATOR> Base;


    BasicImageIterator(ITERATOR line)
    : Base(line)
    {}

    BasicImageIterator()
    : Base()
    {}
};

/********************************************************/
/*                                                      */
/*                ConstBasicImageIterator               */
/*                                                      */
/********************************************************/

/** Implementation of the standard const image iterator for \ref vigra::BasicImage.
    See \ref vigra::ConstImageIterator for documentation.

    <b>\#include</b> "<a href="basicimage_8hxx-source.html">vigra/basicimage.hxx</a>"
    Namespace: vigra
*/
template <class PIXELTYPE, class ITERATOR>
class ConstBasicImageIterator
: public BasicImageIteratorBase<ConstBasicImageIterator<PIXELTYPE, ITERATOR>,
                    PIXELTYPE, PIXELTYPE const &, PIXELTYPE const *, ITERATOR>
{
  public:

    typedef BasicImageIteratorBase<ConstBasicImageIterator,
              PIXELTYPE, PIXELTYPE const &, PIXELTYPE const *, ITERATOR> Base;


    ConstBasicImageIterator(ITERATOR line)
    : Base(line)
    {}

    ConstBasicImageIterator(BasicImageIterator<PIXELTYPE, ITERATOR> const & rhs)
    : Base(rhs.x, rhs.y)
    {}

    ConstBasicImageIterator()
    : Base()
    {}

    ConstBasicImageIterator &
    operator=(BasicImageIterator<PIXELTYPE, ITERATOR> const & rhs)
    {
        Base::x = rhs.x;
        Base::y = rhs.y;
        return *this;
    }

};

/********************************************************/
/*                                                      */
/*             definition of iterator traits            */
/*                                                      */
/********************************************************/


#ifndef NO_PARTIAL_TEMPLATE_SPECIALIZATION

template <class T>
struct IteratorTraits<BasicImageIterator<T, T**> >
: public IteratorTraitsBase<BasicImageIterator<T, T**> >
{
    typedef typename AccessorTraits<T>::default_accessor  DefaultAccessor;
    typedef DefaultAccessor                               default_accessor;
};

template <class T>
struct IteratorTraits<ConstBasicImageIterator<T, T**> >
: public IteratorTraitsBase<ConstBasicImageIterator<T, T**> >
{
    typedef typename AccessorTraits<T>::default_const_accessor  DefaultAccessor;
    typedef DefaultAccessor                               default_accessor;
};

#else // NO_PARTIAL_TEMPLATE_SPECIALIZATION

#define VIGRA_DEFINE_ITERATORTRAITS(VALUETYPE) \
    template <>  \
    struct IteratorTraits<BasicImageIterator<VALUETYPE, VALUETYPE **> > \
    : public IteratorTraitsBase<BasicImageIterator<VALUETYPE, VALUETYPE **> > \
    { \
        typedef typename AccessorTraits<VALUETYPE >::default_accessor  DefaultAccessor; \
        typedef DefaultAccessor                               default_accessor; \
    }; \
    \
    template <>  \
    struct IteratorTraits<ConstBasicImageIterator<VALUETYPE, VALUETYPE **> > \
    : public IteratorTraitsBase<ConstBasicImageIterator<VALUETYPE, VALUETYPE **> > \
    { \
        typedef typename AccessorTraits<VALUETYPE >::default_const_accessor  DefaultAccessor; \
        typedef DefaultAccessor                               default_accessor; \
    };

VIGRA_DEFINE_ITERATORTRAITS(RGBValue<unsigned char>)
VIGRA_DEFINE_ITERATORTRAITS(RGBValue<short>)
VIGRA_DEFINE_ITERATORTRAITS(RGBValue<int>)
VIGRA_DEFINE_ITERATORTRAITS(RGBValue<float>)
VIGRA_DEFINE_ITERATORTRAITS(RGBValue<double>)

#define VIGRA_PIXELTYPE TinyVector<unsigned char, 2>
VIGRA_DEFINE_ITERATORTRAITS(VIGRA_PIXELTYPE)
#undef VIGRA_PIXELTYPE
#define VIGRA_PIXELTYPE TinyVector<unsigned char, 3>
VIGRA_DEFINE_ITERATORTRAITS(VIGRA_PIXELTYPE)
#undef VIGRA_PIXELTYPE
#define VIGRA_PIXELTYPE TinyVector<unsigned char, 4>
VIGRA_DEFINE_ITERATORTRAITS(VIGRA_PIXELTYPE)
#undef VIGRA_PIXELTYPE
#define VIGRA_PIXELTYPE TinyVector<short, 2>
VIGRA_DEFINE_ITERATORTRAITS(VIGRA_PIXELTYPE)
#undef VIGRA_PIXELTYPE
#define VIGRA_PIXELTYPE TinyVector<short, 3>
VIGRA_DEFINE_ITERATORTRAITS(VIGRA_PIXELTYPE)
#undef VIGRA_PIXELTYPE
#define VIGRA_PIXELTYPE TinyVector<short, 4>
VIGRA_DEFINE_ITERATORTRAITS(VIGRA_PIXELTYPE)
#undef VIGRA_PIXELTYPE
#define VIGRA_PIXELTYPE TinyVector<int, 2>
VIGRA_DEFINE_ITERATORTRAITS(VIGRA_PIXELTYPE)
#undef VIGRA_PIXELTYPE
#define VIGRA_PIXELTYPE TinyVector<int, 3>
VIGRA_DEFINE_ITERATORTRAITS(VIGRA_PIXELTYPE)
#undef VIGRA_PIXELTYPE
#define VIGRA_PIXELTYPE TinyVector<int, 4>
VIGRA_DEFINE_ITERATORTRAITS(VIGRA_PIXELTYPE)
#undef VIGRA_PIXELTYPE
#define VIGRA_PIXELTYPE TinyVector<float, 2>
VIGRA_DEFINE_ITERATORTRAITS(VIGRA_PIXELTYPE)
#undef VIGRA_PIXELTYPE
#define VIGRA_PIXELTYPE TinyVector<float, 3>
VIGRA_DEFINE_ITERATORTRAITS(VIGRA_PIXELTYPE)
#undef VIGRA_PIXELTYPE
#define VIGRA_PIXELTYPE TinyVector<float, 4>
VIGRA_DEFINE_ITERATORTRAITS(VIGRA_PIXELTYPE)
#undef VIGRA_PIXELTYPE
#define VIGRA_PIXELTYPE TinyVector<double, 2>
VIGRA_DEFINE_ITERATORTRAITS(VIGRA_PIXELTYPE)
#undef VIGRA_PIXELTYPE
#define VIGRA_PIXELTYPE TinyVector<double, 3>
VIGRA_DEFINE_ITERATORTRAITS(VIGRA_PIXELTYPE)
#undef VIGRA_PIXELTYPE
#define VIGRA_PIXELTYPE TinyVector<double, 4>
VIGRA_DEFINE_ITERATORTRAITS(VIGRA_PIXELTYPE)
#undef VIGRA_PIXELTYPE

#undef VIGRA_DEFINE_ITERATORTRAITS

#endif // NO_PARTIAL_TEMPLATE_SPECIALIZATION

/********************************************************/
/*                                                      */
/*                     BasicImage                       */
/*                                                      */
/********************************************************/

/** \brief Fundamental class template for images.

    A customized memory allocator can be specified as a templated argument
    ans passed in the constructor.

    <b>\#include</b> "<a href="basicimage_8hxx-source.html">vigra/basicimage.hxx</a>"

    Namespace: vigra
*/
template <class PIXELTYPE, class Alloc = std::allocator<PIXELTYPE> >
class BasicImage
{
  public:

        /** the BasicImage's pixel type
        */
    typedef PIXELTYPE value_type;

        /** the BasicImage's pixel type
        */
    typedef PIXELTYPE PixelType;

        /** the BasicImage's reference type (i.e. the
            return type of <TT>image[diff]</TT> and <TT>image(dx,dy)</TT>)
        */
    typedef PIXELTYPE &       reference;

        /** the BasicImage's const reference type (i.e. the
            return type of <TT>image[diff]</TT> and <TT>image(dx,dy)</TT>
            when <TT>image</TT> is const)
        */
    typedef PIXELTYPE const & const_reference;

        /** the BasicImage's pointer type
        */
    typedef PIXELTYPE *       pointer;

        /** the BasicImage's const pointer type
        */
    typedef PIXELTYPE const * const_pointer;

        /** the BasicImage's 1D random access iterator
            (note: lower case 'iterator' is a STL compatible 1D random
             access iterator, don't confuse with capitalized Iterator)
        */
    typedef PIXELTYPE * iterator;

        /** deprecated, use <TT>iterator</TT> instead
        */
    typedef PIXELTYPE * ScanOrderIterator;

        /** the BasicImage's 1D random access const iterator
            (note: lower case 'const_iterator' is a STL compatible 1D
            random access const iterator)
        */
    typedef PIXELTYPE const * const_iterator;

        /** deprecated, use <TT>const_iterator</TT> instead
        */
    typedef PIXELTYPE const * ConstScanOrderIterator;

        /** the BasicImage's 2D random access iterator ('traverser')
        */
    typedef BasicImageIterator<PIXELTYPE, PIXELTYPE **> traverser;

        /** deprecated, use <TT>traverser</TT> instead
        */
    typedef BasicImageIterator<PIXELTYPE, PIXELTYPE **> Iterator;

        /** the BasicImage's 2D random access const iterator ('const traverser')
        */
    typedef
        ConstBasicImageIterator<PIXELTYPE, PIXELTYPE **>
        const_traverser;

        /** deprecated, use <TT>const_traverser</TT> instead
        */
    typedef
        ConstBasicImageIterator<PIXELTYPE, PIXELTYPE **>
        ConstIterator;

        /** the BasicImage's difference type (argument type of image[diff])
        */
    typedef Diff2D difference_type;

         /** the BasicImage's size type (result type of image.size())
        */
    typedef Size2D size_type;

       /** the BasicImage's default accessor
        */
    typedef typename
          IteratorTraits<traverser>::DefaultAccessor Accessor;

        /** the BasicImage's default const accessor
        */
    typedef typename
          IteratorTraits<const_traverser>::DefaultAccessor ConstAccessor;

        /** the BasicImage's allocator (default: std::allocator<value_type>)
        */
    typedef Alloc allocator_type;

    typedef Alloc Allocator;
    typedef typename Alloc::template rebind<PIXELTYPE *>::other LineAllocator;

        /** construct image of size 0x0
        */
    BasicImage()
    : data_(0),
      width_(0),
      height_(0)
    {}

        /** construct image of size 0x0, use the specified allocator.
        */
    explicit BasicImage(Alloc const & alloc)
    : data_(0),
      width_(0),
      height_(0),
      allocator_(alloc),
      pallocator_(alloc)
    {}

        /** construct image of size width x height, use the specified allocator.
        */
    BasicImage(int width, int height, Alloc const & alloc = Alloc())
    : data_(0),
      width_(0),
      height_(0),
      allocator_(alloc),
      pallocator_(alloc)
    {
        vigra_precondition((width >= 0) && (height >= 0),
             "BasicImage::BasicImage(int width, int height): "
             "width and height must be >= 0.\n");

        resize(width, height, value_type());
    }

        /** construct image of size size.x x size.y, use the specified allocator.
        */
    explicit BasicImage(difference_type const & size, Alloc const & alloc = Alloc())
    : data_(0),
      width_(0),
      height_(0),
      allocator_(alloc),
      pallocator_(alloc)
    {
        vigra_precondition((size.x >= 0) && (size.y >= 0),
             "BasicImage::BasicImage(Diff2D size): "
             "size.x and size.y must be >= 0.\n");

        resize(size.x, size.y, value_type());
    }

        /** construct image of size width*height and initialize every
        pixel with the value \a d (use this constructor, if
        value_type doesn't have a default constructor). 
        Use the specified allocator.
        */
    BasicImage(int width, int height, value_type const & d, Alloc const & alloc = Alloc())
    : data_(0),
      width_(0),
      height_(0),
      allocator_(alloc),
      pallocator_(alloc)
    {
        vigra_precondition((width >= 0) && (height >= 0),
             "BasicImage::BasicImage(int width, int height, value_type const & ): "
             "width and height must be >= 0.\n");

        resize(width, height, d);
    }

        /** construct image of size size.x x size.y and initialize
        every pixel with given data (use this constructor, if
        value_type doesn't have a default constructor). Use the specified allocator.
        */
    explicit BasicImage(difference_type const & size, value_type const & d, Alloc const & alloc = Alloc())
    : data_(0),
      width_(0),
      height_(0),
      allocator_(alloc),
      pallocator_(alloc)
    {
        vigra_precondition((size.x >= 0) && (size.y >= 0),
             "BasicImage::BasicImage(Diff2D const & size, value_type const & v): "
             "size.x and size.y must be >= 0.\n");

        resize(size.x, size.y, d);
    }


        /** construct image of size width*height and copy the data from the
            given C-style array \a d. Use the specified allocator.
        */
    BasicImage(int width, int height, const_pointer d, Alloc const & alloc = Alloc())
    : data_(0),
      width_(0),
      height_(0),
      allocator_(alloc),
      pallocator_(alloc)
    {
        vigra_precondition((width >= 0) && (height >= 0),
             "BasicImage::BasicImage(int width, int height, const_pointer ): "
             "width and height must be >= 0.\n");

        resizeCopy(width, height, d);
    }

        /** construct image of size size.x x size.y  and copy the data from the
            given C-style array. Use the specified allocator.
        */
    explicit BasicImage(difference_type const & size, const_pointer d, Alloc const & alloc = Alloc())
    : data_(0),
      width_(0),
      height_(0),
      allocator_(alloc),
      pallocator_(alloc)
    {
        vigra_precondition((size.x >= 0) && (size.y >= 0),
             "BasicImage::BasicImage(Diff2D const & size, const_pointer): "
             "size.x and size.y must be >= 0.\n");

        resizeCopy(size.x, size.y, d);
    }

        /** copy rhs image
        */
    BasicImage(const BasicImage & rhs)
    : data_(0),
      width_(0),
      height_(0),
      allocator_(rhs.allocator_),
      pallocator_(rhs.pallocator_)
    {
        resizeCopy(rhs);
    }

        /** destructor
        */
    ~BasicImage()
    {
        deallocate();
    }

        /** copy rhs image (image is resized if necessary)
        */
    BasicImage & operator=(const BasicImage & rhs);

        /** \deprecated set Image with const value
        */
    BasicImage & operator=(value_type pixel);

        /** set Image with const value
        */
    BasicImage & init(value_type const & pixel);

        /** reset image to specified size (dimensions must not be negative)
            (old data are kept if new size matches old size)
        */
    void resize(int width, int height)
    {
        if(width != width_ || height != height_)
            resize(width, height, value_type());
    }

        /** reset image to specified size (dimensions must not be negative)
            (old data are kept if new size matches old size)
        */
    void resize(difference_type const & size)
    {
        if(size.x != width_ || size.y != height_)
        {
            resize(size.x, size.y, value_type());
        }
    }

        /** reset image to specified size and initialize it with
            given data (use this if value_type doesn't have a default
            constructor, dimensions must not be negative,
            old data are kept if new size matches old size)
        */
    void resize(int width, int height, value_type const & d);

        /** resize image to given size and initialize by copying data
            from the C-style arra \a data.
        */
    void resizeCopy(int width, int height, const_pointer data);

        /** resize image to size of other image and copy it's data
        */
    void resizeCopy(const BasicImage & rhs)
    {
        resizeCopy(rhs.width(), rhs.height(), rhs.data_);
    }

        /** swap the internal data with the rhs image in constant time
        */
    void swap( BasicImage & rhs );

        /** width of Image
        */
    int width() const
    {
        return width_;
    }

        /** height of Image
        */
    int height() const
    {
        return height_;
    }

        /** size of Image
        */
    size_type size() const
    {
        return size_type(width(), height());
    }

        /** test whether a given coordinate is inside the image
        */
    bool isInside(difference_type const & d) const
    {
        return d.x >= 0 && d.y >= 0 &&
               d.x < width() && d.y < height();
    }

        /** access pixel at given location. <br>
	    usage: <TT> value_type value = image[Diff2D(1,2)] </TT>
        */
    reference operator[](difference_type const & d)
    {
        return lines_[d.y][d.x];
    }

        /** read pixel at given location. <br>
	    usage: <TT> value_type value = image[Diff2D(1,2)] </TT>
        */
    const_reference operator[](difference_type const & d) const
    {
        return lines_[d.y][d.x];
    }

        /** access pixel at given location. <br>
	    usage: <TT> value_type value = image(1,2) </TT>
        */
    reference operator()(int dx, int dy)
    {
        return lines_[dy][dx];
    }

        /** read pixel at given location. <br>
	    usage: <TT> value_type value = image(1,2) </TT>
        */
    const_reference operator()(int dx, int dy) const
    {
        return lines_[dy][dx];
    }

        /** access pixel at given location.
	        Note that the 'x' index is the trailing index. <br>
	    usage: <TT> value_type value = image[2][1] </TT>
        */
    pointer operator[](int dy)
    {
        return lines_[dy];
    }

        /** read pixel at given location.
	        Note that the 'x' index is the trailing index. <br>
	    usage: <TT> value_type value = image[2][1] </TT>
        */
    const_pointer operator[](int dy) const
    {
        return lines_[dy];
    }

        /** init 2D random access iterator poining to upper left pixel
        */
    traverser upperLeft()
    {
        vigra_precondition(data_ != 0,
          "BasicImage::upperLeft(): image must have non-zero size.");
        return traverser(lines_);
    }

        /** init 2D random access iterator poining to
         pixel(width, height), i.e. one pixel right and below lower right
         corner of the image as is common in C/C++.
        */
    traverser lowerRight()
    {
        vigra_precondition(data_ != 0,
          "BasicImage::lowerRight(): image must have non-zero size.");
        return upperLeft() + size();
    }

        /** init 2D random access const iterator poining to upper left pixel
        */
    const_traverser upperLeft() const
    {
        vigra_precondition(data_ != 0,
          "BasicImage::upperLeft(): image must have non-zero size.");
        return const_traverser(const_cast<PIXELTYPE **>(lines_));
    }

        /** init 2D random access const iterator poining to
         pixel(width, height), i.e. one pixel right and below lower right
         corner of the image as is common in C/C++.
        */
    const_traverser lowerRight() const
    {
        vigra_precondition(data_ != 0,
          "BasicImage::lowerRight(): image must have non-zero size.");
        return upperLeft() + size();
    }

        /** init 1D random access iterator pointing to first pixel
        */
    iterator begin()
    {
        vigra_precondition(data_ != 0,
          "BasicImage::begin(): image must have non-zero size.");
        return data_;
    }

        /** init 1D random access iterator pointing past the end
        */
    iterator end()
    {
        vigra_precondition(data_ != 0,
          "BasicImage::end(): image must have non-zero size.");
        return data_ + width() * height();
    }

        /** init 1D random access const iterator pointing to first pixel
        */
    const_iterator begin() const
    {
        vigra_precondition(data_ != 0,
          "BasicImage::begin(): image must have non-zero size.");
        return data_;
    }

        /** init 1D random access const iterator pointing past the end
        */
    const_iterator end() const
    {
        vigra_precondition(data_ != 0,
          "BasicImage::end(): image must have non-zero size.");
        return data_ + width() * height();
    }

        /** return default accessor
        */
    Accessor accessor()
    {
        return Accessor();
    }

        /** return default const accessor
        */
    ConstAccessor accessor() const
    {
        return ConstAccessor();
    }

  private:

    void deallocate();

    value_type ** initLineStartArray(value_type * data, int width, int height);

    PIXELTYPE * data_;
    PIXELTYPE ** lines_;
    int width_, height_;
    Alloc allocator_;
    LineAllocator pallocator_;
};

template <class PIXELTYPE, class Alloc>
BasicImage<PIXELTYPE, Alloc> &
BasicImage<PIXELTYPE, Alloc>::operator=(const BasicImage<PIXELTYPE, Alloc> & rhs)
{
    if(this != &rhs)
    {
        if((width() != rhs.width()) ||
           (height() != rhs.height()))
        {
            resizeCopy(rhs);
        }
        else
        {
            ConstScanOrderIterator is = rhs.begin();
            ConstScanOrderIterator iend = rhs.end();
            ScanOrderIterator id = begin();

            for(; is != iend; ++is, ++id) *id = *is;
        }
    }
    return *this;
}

template <class PIXELTYPE, class Alloc>
BasicImage<PIXELTYPE, Alloc> &
BasicImage<PIXELTYPE, Alloc>::operator=(value_type pixel)
{
    ScanOrderIterator i = begin();
    ScanOrderIterator iend = end();

    for(; i != iend; ++i) *i = pixel;

    return *this;
}

template <class PIXELTYPE, class Alloc>
BasicImage<PIXELTYPE, Alloc> &
BasicImage<PIXELTYPE, Alloc>::init(value_type const & pixel)
{
    ScanOrderIterator i = begin();
    ScanOrderIterator iend = end();

    for(; i != iend; ++i) *i = pixel;

    return *this;
}

template <class PIXELTYPE, class Alloc>
void
BasicImage<PIXELTYPE, Alloc>::resize(int width, int height, value_type const & d)
{
    vigra_precondition((width >= 0) && (height >= 0),
         "BasicImage::resize(int width, int height, value_type const &): "
         "width and height must be >= 0.\n");

    if (width_ != width || height_ != height)  // change size?
    {
        value_type * newdata = 0;
        value_type ** newlines = 0;
        if(width*height > 0)
        {
            if (width*height != width_*height_) // different sizes, must reallocate
            {
                newdata = allocator_.allocate(width*height);
                std::uninitialized_fill_n(newdata, width*height, d);
                newlines = initLineStartArray(newdata, width, height);
                deallocate();
            }
            else // need only to reshape
            {
                newdata = data_;
                std::fill_n(newdata, width*height, d);
                newlines = initLineStartArray(newdata, width, height);
                pallocator_.deallocate(lines_, height_);
            }
        }
        else
        {
            deallocate();
        }

        data_ = newdata;
        lines_ = newlines;
        width_ = width;
        height_ = height;
    }
    else if(width*height > 0) // keep size, re-init data
    {
        std::fill_n(data_, width*height, d);
    }
}


template <class PIXELTYPE, class Alloc>
void
BasicImage<PIXELTYPE, Alloc>::resizeCopy(int width, int height, const_pointer data)
{
    int newsize = width*height;
    if (width_ != width || height_ != height)  // change size?
    {
        value_type * newdata = 0;
        value_type ** newlines = 0;
        if(newsize > 0)
        {
            if (newsize != width_*height_) // different sizes, must reallocate
            {
                newdata = allocator_.allocate(newsize);
                std::uninitialized_copy(data, data + newsize, newdata);
                newlines = initLineStartArray(newdata, width, height);
                deallocate();
            }
            else // need only to reshape
            {
                newdata = data_;
                std::copy(data, data + newsize, newdata);
                newlines = initLineStartArray(newdata, width, height);
                pallocator_.deallocate(lines_, height_);
            }
        }
        else
        {
            deallocate();
        }

        data_ = newdata;
        lines_ = newlines;
        width_ = width;
        height_ = height;
    }
    else if(newsize > 0) // keep size, copy data
    {
        std::copy(data, data + newsize, data_);
    }
}

template <class PIXELTYPE, class Alloc>
void
BasicImage<PIXELTYPE, Alloc>::swap( BasicImage<PIXELTYPE, Alloc>& rhs )
{
  if (&rhs!=this)
  {
    std::swap( data_, rhs.data_ );
    std::swap( lines_, rhs.lines_ );
    std::swap( width_, rhs.width_ );
    std::swap( height_, rhs.height_ );
  }
}

template <class PIXELTYPE, class Alloc>
void
BasicImage<PIXELTYPE, Alloc>::deallocate()
{
    if(data_)
    {
        ScanOrderIterator i = begin();
        ScanOrderIterator iend = end();

        for(; i != iend; ++i)   (*i).~PIXELTYPE();

        allocator_.deallocate(data_, width()*height());
        pallocator_.deallocate(lines_, height_);
    }
}

template <class PIXELTYPE, class Alloc>
PIXELTYPE **
BasicImage<PIXELTYPE, Alloc>::initLineStartArray(value_type * data, int width, int height)
{
    value_type ** lines = pallocator_.allocate(height);
    for(int y=0; y<height; ++y)
         lines[y] = data + y*width;
    return lines;
}

/********************************************************/
/*                                                      */
/*              argument object factories               */
/*                                                      */
/********************************************************/

template <class PixelType, class Accessor, class Alloc>
inline triple<typename BasicImage<PixelType, Alloc>::const_traverser,
              typename BasicImage<PixelType, Alloc>::const_traverser, Accessor>
srcImageRange(BasicImage<PixelType, Alloc> const & img, Accessor a)
{
    return triple<typename BasicImage<PixelType, Alloc>::const_traverser,
                  typename BasicImage<PixelType, Alloc>::const_traverser,
          Accessor>(img.upperLeft(),
                    img.lowerRight(),
                    a);
}

template <class PixelType, class Accessor, class Alloc>
inline pair<typename BasicImage<PixelType, Alloc>::const_traverser, Accessor>
srcImage(BasicImage<PixelType, Alloc> const & img, Accessor a)
{
    return pair<typename BasicImage<PixelType, Alloc>::const_traverser,
                Accessor>(img.upperLeft(), a);
}

template <class PixelType, class Accessor, class Alloc>
inline triple<typename BasicImage<PixelType, Alloc>::traverser,
              typename BasicImage<PixelType, Alloc>::traverser, Accessor>
destImageRange(BasicImage<PixelType, Alloc> & img, Accessor a)
{
    return triple<typename BasicImage<PixelType, Alloc>::traverser,
                  typename BasicImage<PixelType, Alloc>::traverser,
          Accessor>(img.upperLeft(),
                    img.lowerRight(),
                    a);
}

template <class PixelType, class Accessor, class Alloc>
inline pair<typename BasicImage<PixelType, Alloc>::traverser, Accessor>
destImage(BasicImage<PixelType, Alloc> & img, Accessor a)
{
    return pair<typename BasicImage<PixelType, Alloc>::traverser,
                Accessor>(img.upperLeft(), a);
}

template <class PixelType, class Accessor, class Alloc>
inline pair<typename BasicImage<PixelType, Alloc>::const_traverser, Accessor>
maskImage(BasicImage<PixelType, Alloc> const & img, Accessor a)
{
    return pair<typename BasicImage<PixelType, Alloc>::const_traverser,
                Accessor>(img.upperLeft(), a);
}

/****************************************************************/

template <class PixelType, class Alloc>
inline triple<typename BasicImage<PixelType, Alloc>::const_traverser,
              typename BasicImage<PixelType, Alloc>::const_traverser,
              typename BasicImage<PixelType, Alloc>::ConstAccessor>
srcImageRange(BasicImage<PixelType, Alloc> const & img)
{
    return triple<typename BasicImage<PixelType, Alloc>::const_traverser,
                  typename BasicImage<PixelType, Alloc>::const_traverser,
                  typename BasicImage<PixelType, Alloc>::ConstAccessor>(img.upperLeft(),
                                                                 img.lowerRight(),
                                                                 img.accessor());
}

template <class PixelType, class Alloc>
inline pair< typename BasicImage<PixelType, Alloc>::const_traverser,
             typename BasicImage<PixelType, Alloc>::ConstAccessor>
srcImage(BasicImage<PixelType, Alloc> const & img)
{
    return pair<typename BasicImage<PixelType, Alloc>::const_traverser,
                typename BasicImage<PixelType, Alloc>::ConstAccessor>(img.upperLeft(),
                                                               img.accessor());
}

template <class PixelType, class Alloc>
inline triple< typename BasicImage<PixelType, Alloc>::traverser,
               typename BasicImage<PixelType, Alloc>::traverser,
               typename BasicImage<PixelType, Alloc>::Accessor>
destImageRange(BasicImage<PixelType, Alloc> & img)
{
    return triple<typename BasicImage<PixelType, Alloc>::traverser,
                  typename BasicImage<PixelType, Alloc>::traverser,
                  typename BasicImage<PixelType, Alloc>::Accessor>(img.upperLeft(),
                                                            img.lowerRight(),
                                                            img.accessor());
}

template <class PixelType, class Alloc>
inline pair< typename BasicImage<PixelType, Alloc>::traverser,
             typename BasicImage<PixelType, Alloc>::Accessor>
destImage(BasicImage<PixelType, Alloc> & img)
{
    return pair<typename BasicImage<PixelType, Alloc>::traverser,
                typename BasicImage<PixelType, Alloc>::Accessor>(img.upperLeft(),
                                                          img.accessor());
}

template <class PixelType, class Alloc>
inline pair< typename BasicImage<PixelType, Alloc>::const_traverser,
             typename BasicImage<PixelType, Alloc>::ConstAccessor>
maskImage(BasicImage<PixelType, Alloc> const & img)
{
    return pair<typename BasicImage<PixelType, Alloc>::const_traverser,
                typename BasicImage<PixelType, Alloc>::ConstAccessor>(img.upperLeft(),
                                                               img.accessor());
}

} // namespace vigra

#endif // VIGRA_BASICIMAGE_HXX
