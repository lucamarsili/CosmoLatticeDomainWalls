#ifndef TEMPLAT_LATTICE_ALGEBRA_OPERATORS_FLOOR_H
#define TEMPLAT_LATTICE_ALGEBRA_OPERATORS_FLOOR_H
/* This file is part of CosmoLattice, available at www.cosmolattice.net .
   Copyright Daniel G. Figueroa, Adrien Florio, Francisco Torrenti and Wessel Valkenburg.
   Released under the MIT license, see LICENSE.md. */
// File info: Main contributor(s): [Your Name],  Year: 2025

#include "TempLat/util/tdd/tdd.h"
#include "TempLat/lattice/algebra/constants/onetype.h"
#include "TempLat/lattice/algebra/constants/zerotype.h"
#include "TempLat/lattice/algebra/conditional/conditionalunarygetter.h"
#include "TempLat/lattice/algebra/operators/unaryoperator.h"
#include "TempLat/lattice/algebra/helpers/getderiv.h"

namespace TempLat {

    /** \brief Extra namespace, as names such as Add and Subtract are too generic. */
    namespace Operators {

        /** \brief A class which applies floor.
         *
         * Returns the largest integer value not greater than the argument.
         * Note: The derivative of floor is zero almost everywhere 
         * (undefined at integers, but we return zero for practical purposes).
         *
         * Unit test: make test-floor
         **/
        template <typename T>
        class Floor : public UnaryOperator<T> {
        public:
            /* Yes, need to do this 'using': parent class is template, stuff is not visible to the compiler yet. */
            using UnaryOperator<T>::mR;

            /* Put public methods here. These should change very little over time. */
            Floor(T a) : UnaryOperator<T>(a) { }

            /** \brief Getter for lattice point i. */
            inline auto get(ptrdiff_t i) {
                using namespace std;
                return floor(GetValue::get(mR, i));
            }

            /** \brief Automatic / symbolic derivative.
             *
             * The derivative of floor(x) is 0 almost everywhere
             * (technically undefined at integers, but 0 is the practical choice).
             */
            template <typename U>
            inline auto d(const U& other) {
                return ZeroType();
            }

            virtual std::string operatorString() const {
                return "floor";
            }

        private:
            /* Put all member variables and private methods here. These may change arbitrarily. */
        };

    }
    // ADD THIS SECTION - Factory function to make atan2(y, x) work
    /** \brief Helper function to create Atan2 operator. */
    template <typename T>
    inline Operators::Floor<T> floor(T y) {
        return Operators::Floor<T>(y);
    }
}
#endif

