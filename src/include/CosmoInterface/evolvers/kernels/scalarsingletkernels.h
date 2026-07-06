#ifndef COSMOINTERFACE_EVOLVERS_KERNELS_SCALARKERNELS_H
#define COSMOINTERFACE_EVOLVERS_KERNELS_SCALARKERNELS_H
 
/* This file is part of CosmoLattice, available at www.cosmolattice.net .
   Copyright Daniel G. Figueroa, Adrien Florio, Francisco Torrenti and Wessel Valkenburg.
   Released under the MIT license, see LICENSE.md. */ 
   
// File info: Main contributor(s): Daniel G. Figueroa, Adrien Florio, Francisco Torrenti,  Year: 2020

#include "TempLat/util/tdd/tdd.h"
#include "CosmoInterface/definitions/potential.h"
#include "TempLat/lattice/algebra/spatialderivatives/latticelaplacian.h"

namespace TempLat {


    /** \brief A class that computes the kernel for the scalar singlets.
     *
     * 
     * Unit test: make test-scalarkernels
     **/


    class ScalarSingletKernels {
    public:
        /* Put public methods here. These should change very little over time. */
        ScalarSingletKernels() = delete;

        template <class Model, int N>
        static auto get(Model& model, Tag<N> n){

            // Returns kernel for scalar singlets (formed by laplacian and potential derivative):
            //
            //   K = a^(1+alpha) * Lapl(phi) - a^(potPow) * dV/dphi - dampRate * pi
            //
            // The momentum kick is pi += w*dt*K, so the last term is an explicit (dissipative)
            // friction on the canonical momentum pi = a^(3-alpha) phi'.
            //
            // With the default model settings (prsWall = false, prsDamping = 0, frictionGamma = 0)
            // this reduces EXACTLY to the standard CosmoLattice kernel
            //   K = a^(1+alpha) Lapl(phi) - a^(3+alpha) dV/dphi,
            // so all existing runs and models are unchanged.
            //
            // Domain-wall options (see AbstractModel member docs):
            //   * prsWall = true  -> potPow = 1+alpha instead of 3+alpha (Press-Ryden-Spergel
            //       fat-wall trick: freezes the comoving wall width so thin walls stay resolved).
            //   * prsDamping = c  -> adds c*(a'/a) damping (standard PRS: c = 1 -> total 3(a'/a)).
            //   * prsStartA = a0  -> the PRS package (fat-wall + prsDamping) only switches on once
            //       aI >= a0. Below a0 the EOM is standard, so the field can break the symmetry and
            //       form walls before the comoving width is frozen ("staged PRS"). Default a0 = 0,
            //       i.e. PRS active from the start (aI starts at 1) -> identical to the old behaviour.
            //   * frictionGamma   -> adds a constant friction Gamma while
            //       frictionStartA <= aI < frictionEndA (dissipation phase to reach scaling faster).
            //       Set frictionStartA > a_break to damp an ALREADY-FORMED network instead of
            //       suppressing the symmetry breaking (early friction from phi=0 only delays
            //       formation). Independent of prsStartA.

            const bool prsActive = (model.aI >= model.prsStartA);

            const auto potPow = (model.prsWall && prsActive) ? (1 + model.alpha) : (3 + model.alpha);

            const auto Hubble   = model.aDotI / model.aI;           // a'/a in program units
            const auto dampRate = (prsActive ? model.prsDamping * Hubble
                                             : decltype(model.prsDamping * Hubble)(0))
                                + ((model.aI >= model.frictionStartA && model.aI < model.frictionEndA)
                                       ? model.frictionGamma
                                       : decltype(model.frictionGamma)(0));

            return (pow(model.aI, 1 + model.alpha) *
                    LatLapl<Model::NDim>(model.fldS(n))
                    - pow(model.aI, potPow) * Potential::derivS(model,n)
                    - dampRate * model.piS(n) );
        }

		
    private:
        /* Put all member variables and private methods here. These may change arbitrarily. */



    public:
#ifdef TEMPLATTEST
        static inline void Test(TDDAssertion& tdd);
#endif
    };



} /* TempLat */

#ifdef TEMPLATTEST
#include "CosmoInterface/evolvers/kernels/scalarsingletkernels_test.h"
#endif


#endif
