#ifndef TEMPLAT_LATTICE_ALGEBRA_OPERATORS_ARCTANGENT_H
#define TEMPLAT_LATTICE_ALGEBRA_OPERATORS_ARCTANGENT_H
/* This file is part of CosmoLattice, available at www.cosmolattice.net .
   Copyright Daniel G. Figueroa, Adrien Florio, Francisco Torrenti and Wessel Valkenburg.
   Released under the MIT license, see LICENSE.md. */
// File info: Main contributor(s): [Your Name],  Year: 2025

#include "TempLat/util/tdd/tdd.h"
#include "TempLat/lattice/algebra/constants/onetype.h"
#include "TempLat/lattice/algebra/constants/zerotype.h"
#include "TempLat/lattice/algebra/conditional/conditionalbinarygetter.h"
#include "TempLat/lattice/algebra/operators/binaryoperator.h"
#include "TempLat/lattice/algebra/operators/multiply.h"
#include "TempLat/lattice/algebra/operators/add.h"
#include "TempLat/lattice/algebra/operators/divide.h"
#include "TempLat/lattice/algebra/helpers/getderiv.h"

namespace TempLat {

    /** \brief Extra namespace, as names such as Add and Subtract are too generic. */
    namespace Operators {

        /** \brief A class which applies atan2(y, x).
         *
         * atan2 is a binary operator that computes the arc tangent of y/x,
         * using the signs of both arguments to determine the quadrant.
         *
         * Unit test: make test-arctangent
         **/
        template <typename T, typename U>
        class Atan2 : public BinaryOperator<T, U> {
        public:
            /* Parent class is template, need 'using' to make members visible. */
            using BinaryOperator<T, U>::mR;  // y argument
            using BinaryOperator<T, U>::mT;  // x argument

            /* Put public methods here. These should change very little over time. */
            Atan2(T y, U x) : BinaryOperator<T, U>(y, x) { }

            /** \brief Getter for lattice point i. */
            inline auto get(ptrdiff_t i) {
                using namespace std;
                return atan2(GetValue::get(mT, i), GetValue::get(mR, i));
            }

            /**Automatic / symbolic derivative.
             *
             * d/dt atan2(y,x) = (x * dy/dt - y * dx/dt) / (x^2 + y^2)
             */
           

            virtual std::string operatorString() const {
                return "atan2";
            }

        private:
            /* Put all member variables and private methods here. These may change arbitrarily. */
        };

    }
    // ADD THIS SECTION - Factory function to make atan2(y, x) work
    /** \brief Helper function to create Atan2 operator. */
    template <typename T, typename U>
    inline Operators::Atan2<T, U> atan2(T y, U x) {
        return Operators::Atan2<T, U>(y, x);
    }
}
#endif

//Untested