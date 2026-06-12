/*
      This file is part of my_cuda
      Copyright (C) 2021  Julien Thevenon ( julien_thevenon at yahoo.fr )

      This program is free software: you can redistribute it and/or modify
      it under the terms of the GNU General Public License as published by
      the Free Software Foundation, either version 3 of the License, or
      (at your option) any later version.

      This program is distributed in the hope that it will be useful,
      but WITHOUT ANY WARRANTY; without even the implied warranty of
      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
      GNU General Public License for more details.

      You should have received a copy of the GNU General Public License
      along with this program.  If not, see <http://www.gnu.org/licenses/>
*/
#ifndef MY_CUDA_CUDA_STRONG_PRIMITIVE_H
#define MY_CUDA_CUDA_STRONG_PRIMITIVE_H

#include "my_cuda.h"
#include <limits>

namespace my_cuda
{
    template <typename T, typename PHANTOM>
    class CUDA_strong_primitive;

    template <typename T, typename PHANTOM>
    std::ostream &
    operator<<(std::ostream &, const CUDA_strong_primitive<T, PHANTOM> &);

    /**
     * Class used to make difference between differents types using same basic type implementation
     * @tparam T real type used for implementation
     * @tparam PHANTOM type use to discriminate between different strong primitives
     */
    template <typename T, typename PHANTOM>
    class CUDA_strong_primitive
    {
      public:

        template <typename U, typename PHANTOM_U>
        friend
        std::ostream &
        operator<<(std::ostream &
                  ,const CUDA_strong_primitive<U, PHANTOM_U> &
                  );

        typedef T base_type;

        inline
        explicit
        __host__ __device__
        CUDA_strong_primitive()
#ifndef ENABLE_CUDA_CODE
        = default
#endif // ENABLE_CUDA_CODE
        ;

        [[maybe_unused]]
        inline
        explicit
        __host__ __device__
        CUDA_strong_primitive(const T & p_value);

        [[maybe_unused]]
        inline
        explicit
        __host__ __device__
        CUDA_strong_primitive(T && p_value);

        inline
        explicit
        __host__ __device__
        operator T() const;

        inline
        __host__ __device__
        CUDA_strong_primitive<T, PHANTOM> & operator=(T);

        inline
        __host__ __device__
        bool
        operator<(T) const;

        inline
        __host__ __device__
        bool
        operator<(CUDA_strong_primitive<T, PHANTOM>) const;

        inline
        __host__ __device__
        bool
        operator<=(T) const;

        inline
        __host__ __device__
        bool
        operator<=(CUDA_strong_primitive<T, PHANTOM>) const;

        inline
        __host__ __device__
        bool
        operator==(T) const;

        inline
        __host__ __device__
        bool
        operator==(CUDA_strong_primitive<T, PHANTOM>) const;

        inline
        __host__ __device__
        bool
        operator!=(T) const;

        inline
        __host__ __device__
        bool
        operator!=(CUDA_strong_primitive<T, PHANTOM>) const;

        inline
        __host__ __device__
        CUDA_strong_primitive<T, PHANTOM>
        operator+(CUDA_strong_primitive<T, PHANTOM>) const;

        inline
        __host__ __device__
        CUDA_strong_primitive<T, PHANTOM>
        operator+(T) const;

        inline
        __host__ __device__
        CUDA_strong_primitive<T, PHANTOM>
        operator-(CUDA_strong_primitive<T, PHANTOM>) const;

        inline
        __host__ __device__
        CUDA_strong_primitive<T, PHANTOM>
        operator-(T) const;

        inline
        __host__ __device__
        CUDA_strong_primitive<T, PHANTOM>
        operator/(CUDA_strong_primitive<T, PHANTOM>) const;

        inline
        __host__ __device__
        CUDA_strong_primitive<T, PHANTOM>
        operator/(T) const;

        inline
        __host__ __device__
        CUDA_strong_primitive<T, PHANTOM>
        operator*(CUDA_strong_primitive<T, PHANTOM>) const;

        inline
        __host__ __device__
        CUDA_strong_primitive<T, PHANTOM>
        operator*(T) const;

        // prefix increment
        __host__ __device__
        CUDA_strong_primitive<T, PHANTOM> &
        operator++();

      private:
        // Make get private to force the use of an explicit cast
        [[maybe_unused]]
        inline
        __host__ __device__
        const T & get() const;

        [[maybe_unused]]
        inline
        __host__ __device__
        T & get();


        T m_value;
    };

    //-------------------------------------------------------------------------
#ifdef ENABLE_CUDA_CODE
    template <typename T, typename PHANTOM>
    __host__ __device__
    CUDA_strong_primitive<T, PHANTOM>::CUDA_strong_primitive()
    {}
#endif // ENABLE_CUDA_CODE

    //-------------------------------------------------------------------------
    template <typename T, typename PHANTOM>
    [[maybe_unused]]
    __host__ __device__
    CUDA_strong_primitive<T, PHANTOM>::CUDA_strong_primitive(const T & p_value)
    :m_value{p_value}
    {

    }

    //-------------------------------------------------------------------------
    template <typename T,typename PHANTOM>
    [[maybe_unused]]
    __host__ __device__
    CUDA_strong_primitive<T, PHANTOM>::CUDA_strong_primitive(T && p_value)
    :m_value(std::move(p_value))
    {
    }

    //-------------------------------------------------------------------------
    template <typename T, typename PHANTOM>
    [[maybe_unused]]
    __host__ __device__
    const T &
    CUDA_strong_primitive<T, PHANTOM>::get() const
    {
        return m_value;
    }

    //-------------------------------------------------------------------------
    template <typename T, typename PHANTOM>
    [[maybe_unused]]
    __host__ __device__
    T &
    CUDA_strong_primitive<T, PHANTOM>::get()
    {
        return m_value;
    }

    //-------------------------------------------------------------------------
    template <typename T, typename PHANTOM>
    __host__ __device__
    CUDA_strong_primitive<T, PHANTOM>::operator T() const
    {
        return m_value;
    }

    //-------------------------------------------------------------------------
    template <typename T, typename PHANTOM>
    __host__ __device__
    CUDA_strong_primitive<T, PHANTOM> &
    CUDA_strong_primitive<T, PHANTOM>::operator=(T p_value)
    {
        m_value = p_value;
        return *this;
    }

    //-------------------------------------------------------------------------
    template <typename T, typename PHANTOM>
    __host__ __device__
    bool
    CUDA_strong_primitive<T, PHANTOM>::operator<(T p_value) const
    {
        return m_value < p_value;
    }

    //-------------------------------------------------------------------------
    template <typename T, typename PHANTOM>
    __host__ __device__
    bool
    CUDA_strong_primitive<T, PHANTOM>::operator<(CUDA_strong_primitive<T, PHANTOM> p_operand) const
    {
        return m_value < p_operand.m_value;
    }

    //-------------------------------------------------------------------------
    template <typename T, typename PHANTOM>
    __host__ __device__
    bool
    CUDA_strong_primitive<T, PHANTOM>::operator<=(T p_operand) const
    {
        return m_value <= p_operand.m_value;
    }

    //-------------------------------------------------------------------------
    template <typename T, typename PHANTOM>
    __host__ __device__
    bool
    CUDA_strong_primitive<T, PHANTOM>::operator<=(CUDA_strong_primitive<T, PHANTOM> p_operand) const
    {
        return m_value <= p_operand.m_value;
    }

    //-------------------------------------------------------------------------
    template <typename T, typename PHANTOM>
    __host__ __device__
    CUDA_strong_primitive<T, PHANTOM> &
    CUDA_strong_primitive<T, PHANTOM>::operator++()
    {
        ++m_value;
        return *this;
    }

    //-------------------------------------------------------------------------
    template <typename T, typename PHANTOM>
    __host__ __device__
    bool
    CUDA_strong_primitive<T, PHANTOM>::operator==(T p_value) const
    {
        return m_value == p_value;
    }

    //-------------------------------------------------------------------------
    template <typename T, typename PHANTOM>
    __host__ __device__
    bool
    CUDA_strong_primitive<T, PHANTOM>::operator==(CUDA_strong_primitive<T, PHANTOM> p_operand) const
    {
        return m_value == p_operand.m_value;
    }

    //-------------------------------------------------------------------------
    template <typename T, typename PHANTOM>
    __host__ __device__
    bool
    CUDA_strong_primitive<T, PHANTOM>::operator!=(T p_value) const
    {
        return m_value != p_value;
    }

    //-------------------------------------------------------------------------
    template <typename T, typename PHANTOM>
    __host__ __device__
    bool
    CUDA_strong_primitive<T, PHANTOM>::operator!=(CUDA_strong_primitive<T, PHANTOM> p_operand) const
    {
        return m_value != p_operand.m_value;
    }

    //-------------------------------------------------------------------------
    template <typename T, typename PHANTOM>
    __host__ __device__
    CUDA_strong_primitive<T, PHANTOM>
    CUDA_strong_primitive<T, PHANTOM>::operator+(CUDA_strong_primitive<T, PHANTOM> p_operand) const
    {
        return CUDA_strong_primitive<T, PHANTOM>(m_value + p_operand.m_value);
    }

    //-------------------------------------------------------------------------
    template <typename T, typename PHANTOM>
    __host__ __device__
    CUDA_strong_primitive<T, PHANTOM>
    CUDA_strong_primitive<T, PHANTOM>::operator+(T p_value) const
    {
        return CUDA_strong_primitive<T, PHANTOM>(m_value + p_value);
    }

    //-------------------------------------------------------------------------
    template <typename T, typename PHANTOM>
    __host__ __device__
    CUDA_strong_primitive<T, PHANTOM>
    CUDA_strong_primitive<T, PHANTOM>::operator-(CUDA_strong_primitive<T, PHANTOM> p_operand) const
    {
        return CUDA_strong_primitive<T, PHANTOM>(m_value - p_operand.m_value);
    }

    //-------------------------------------------------------------------------
    template <typename T, typename PHANTOM>
    __host__ __device__
    CUDA_strong_primitive<T, PHANTOM>
    CUDA_strong_primitive<T, PHANTOM>::operator-(T p_value) const
    {
        return CUDA_strong_primitive<T, PHANTOM>(m_value - p_value);
    }

    //-------------------------------------------------------------------------
    template <typename T, typename PHANTOM>
    __host__ __device__
    CUDA_strong_primitive<T, PHANTOM>
    CUDA_strong_primitive<T, PHANTOM>::operator/(CUDA_strong_primitive<T, PHANTOM> p_operand) const
    {
        return CUDA_strong_primitive<T, PHANTOM>(m_value / p_operand.m_value);
    }

    //-------------------------------------------------------------------------
    template <typename T, typename PHANTOM>
    __host__ __device__
    CUDA_strong_primitive<T, PHANTOM>
    CUDA_strong_primitive<T, PHANTOM>::operator/(T p_value) const
    {
        return CUDA_strong_primitive<T, PHANTOM>(m_value / p_value);
    }

    //-------------------------------------------------------------------------
    template <typename T, typename PHANTOM>
    __host__ __device__
    CUDA_strong_primitive<T, PHANTOM>
    CUDA_strong_primitive<T, PHANTOM>::operator*(CUDA_strong_primitive<T, PHANTOM> p_operand) const
    {
        return CUDA_strong_primitive<T, PHANTOM>(m_value * p_operand.m_value);
    }

    //-------------------------------------------------------------------------
    template <typename T, typename PHANTOM>
    __host__ __device__
    CUDA_strong_primitive<T, PHANTOM>
    CUDA_strong_primitive<T, PHANTOM>::operator*(T p_value) const
    {
        return CUDA_strong_primitive<T, PHANTOM>(m_value * p_value);
    }

    //-------------------------------------------------------------------------
    template <typename T, typename PHANTOM>
    std::ostream &
    operator<<(std::ostream & p_stream
              ,const CUDA_strong_primitive<T, PHANTOM> & p_operand
              )
    {
        p_stream << p_operand.m_value;
        return p_stream;
    }

}
namespace std
{
    template<typename T, typename PHANTOM>
    struct [[maybe_unused]] is_integral<my_cuda::CUDA_strong_primitive<T, PHANTOM>>
    {
      public:
        static constexpr bool value = is_integral<T>::value;
    };

    template<typename T, typename PHANTOM>
    struct [[maybe_unused]] is_arithmetic<my_cuda::CUDA_strong_primitive<T, PHANTOM>>
    {
      public:
        static constexpr bool value = is_arithmetic<T>::value;
    };

    template<typename T, typename PHANTOM>
    struct [[maybe_unused]] is_scalar<my_cuda::CUDA_strong_primitive<T, PHANTOM>>
    {
      public:
        static constexpr bool value = is_scalar<T>::value;
    };

    template <typename T, typename PHANTOM>
    class is_signed<my_cuda::CUDA_strong_primitive<T, PHANTOM> >
    {
      public:
        static const bool value = is_signed<T>::value;
    };

    template <typename T, typename PHANTOM>
    class make_signed<my_cuda::CUDA_strong_primitive<T, PHANTOM> >
    {
      public:
        typedef my_cuda::CUDA_strong_primitive<typename std::make_signed<T>::type, PHANTOM> type;
    };

    template <typename T, typename PHANTOM>
    class make_unsigned<my_cuda::CUDA_strong_primitive<T, PHANTOM> >
    {
      public:
        typedef my_cuda::CUDA_strong_primitive<typename std::make_unsigned<T>::type, PHANTOM> type;
    };

    template <typename T, typename PHANTOM>
    class numeric_limits<my_cuda::CUDA_strong_primitive<T, PHANTOM> >
    {
      public:

        [[maybe_unused]]
        static
        constexpr bool is_specialized = std::numeric_limits<T>::is_specialized;

        static
        constexpr my_cuda::CUDA_strong_primitive<T, PHANTOM> min() noexcept
        {
            return my_cuda::CUDA_strong_primitive<T, PHANTOM>(std::numeric_limits<T>::min());
        }

        static
        constexpr my_cuda::CUDA_strong_primitive<T, PHANTOM> max() noexcept
        {
            return my_cuda::CUDA_strong_primitive<T, PHANTOM>(std::numeric_limits<T>::max());
        }

        static
        constexpr my_cuda::CUDA_strong_primitive<T, PHANTOM> lowest() noexcept
        {
            return my_cuda::CUDA_strong_primitive<T, PHANTOM>(std::numeric_limits<T>::lowest());
        }

        [[maybe_unused]]
        static
        constexpr int digits = std::numeric_limits<T>::digits;

        [[maybe_unused]]
        static
        constexpr int digits10 = std::numeric_limits<T>::digits10;

        [[maybe_unused]]
        static
        constexpr int max_digits10 = std::numeric_limits<T>::max_digits10;

        [[maybe_unused]]
        static
        constexpr bool is_signed = std::numeric_limits<T>::is_signed;

        [[maybe_unused]]
        static
        constexpr bool is_integer = std::numeric_limits<T>::is_integer;

        [[maybe_unused]]
        static
        constexpr bool is_exact = std::numeric_limits<T>::is_exact;

        [[maybe_unused]]
        static
        constexpr int radix = std::numeric_limits<T>::radix;

        static
        constexpr my_cuda::CUDA_strong_primitive<T, PHANTOM> epsilon() noexcept
        {
            return my_cuda::CUDA_strong_primitive<T, PHANTOM>(std::numeric_limits<T>::epsilon());
        }

        static
        constexpr my_cuda::CUDA_strong_primitive<T, PHANTOM> round_error() noexcept
        {
            return my_cuda::CUDA_strong_primitive<T, PHANTOM>(std::numeric_limits<T>::round_error());
        }

        [[maybe_unused]]
        static
        constexpr int min_exponent = std::numeric_limits<T>::min_exponent;

        [[maybe_unused]]
        static
        constexpr int min_exponent10 = std::numeric_limits<T>::min_exponent10;

        [[maybe_unused]]
        static
        constexpr int max_exponent = std::numeric_limits<T>::max_exponent;

        [[maybe_unused]]
        static
        constexpr int max_exponent10 = std::numeric_limits<T>::max_exponent10;

        [[maybe_unused]]
        static
        constexpr bool has_infinity = std::numeric_limits<T>::has_infinity;

        [[maybe_unused]]
        static
        constexpr bool has_quiet_NaN = std::numeric_limits<T>::has_quiet_NaN;

        [[maybe_unused]]
        static
        constexpr bool has_signaling_NaN = std::numeric_limits<T>::has_signaling_NaN;

        [[maybe_unused]]
        static
        constexpr float_denorm_style has_denorm = std::numeric_limits<T>::has_denorm;

        [[maybe_unused]]
        static
        constexpr bool has_denorm_loss = std::numeric_limits<T>::has_denorm_loss;

        [[maybe_unused]]
        static
        constexpr bool infinity() noexcept { return std::numeric_limits<T>::has_infinity; }

        [[maybe_unused]]
        static
        constexpr bool quiet_NaN() noexcept { return std::numeric_limits<T>::has_quiet_NaN; }

        [[maybe_unused]]
        static
        constexpr bool signaling_NaN() noexcept { return std::numeric_limits<T>::has_signaling_NaN; }

        [[maybe_unused]]
        static
        constexpr bool denorm_min() noexcept { return std::numeric_limits<T>::has_denorm; }

        [[maybe_unused]]
        static
        constexpr bool is_iec559 = std::numeric_limits<T>::is_iec559;

        [[maybe_unused]]
        static
        constexpr bool is_bounded = std::numeric_limits<T>::is_bounded;

        [[maybe_unused]]
        static
        constexpr bool is_modulo = std::numeric_limits<T>::is_modulo;

        [[maybe_unused]]
        static
        constexpr bool traps = std::numeric_limits<T>::traps;

        [[maybe_unused]]
        static
        constexpr bool tinyness_before = std::numeric_limits<T>::tinyness_before;

        [[maybe_unused]]
        static
        constexpr float_round_style round_style = std::numeric_limits<T>::round_style;
    };
}
#endif //MY_CUDA_CUDA_STRONG_PRIMITIVE_H
// EOF