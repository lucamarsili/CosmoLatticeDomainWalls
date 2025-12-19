#ifndef TEMPLAT_LATTICE_ALGEBRA_EQUALTO_H
#define TEMPLAT_LATTICE_ALGEBRA_EQUALTO_H

/*
   A TempLat operator that checks equality pointwise on the lattice
   and returns 1 when equal, 0 otherwise.
*/

#include "TempLat/lattice/algebra/operators/binaryoperator.h"
#include "TempLat/lattice/algebra/helpers/getderiv.h"
#include "TempLat/lattice/algebra/constants/twotype.h"
#include "TempLat/lattice/algebra/constants/onetype.h"
#include "TempLat/lattice/algebra/constants/zerotype.h"

namespace TempLat {

namespace Operators {

    template<typename R, typename T>
    class EqualTo : public BinaryOperator<R,T> {
    public:
        using BinaryOperator<R,T>::mR;
        using BinaryOperator<R,T>::mT;

        EqualTo(const R& pR, const T& pT) :
            BinaryOperator<R,T>(pR, pT) {}

        inline auto get(ptrdiff_t i) {
            auto a = GetValue::get(mR, i);
            auto b = GetValue::get(mT, i);
            return a == b ? 1 : 0;
        }

        virtual std::string operatorString() const {
            return "==";
        }

        template <typename U>
        inline auto d(const U& other) {
            // derivative of comparison is zero
            return ZeroType();
        }
    };

} // namespace Operators

/* Factory: overloaded == operator */
template<typename R, typename T>
inline auto operator==(const R& r, const T& t)
{
    return Operators::EqualTo<R,T>(r, t);
}

} // namespace TempLat

#endif

