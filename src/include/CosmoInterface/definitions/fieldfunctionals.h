#ifndef COSMOINTERFACE_HELPERS_COMPOSITEFIELDS_H
#define COSMOINTERFACE_HELPERS_COMPOSITEFIELDS_H
 
/* This file is part of CosmoLattice, available at www.cosmolattice.net .
   Copyright Daniel G. Figueroa, Adrien Florio, Francisco Torrenti and Wessel Valkenburg.
   Released under the MIT license, see LICENSE.md. */ 
   
// File info: Main contributor(s): Daniel G. Figueroa, Adrien Florio, Francisco Torrenti,  Year: 2020

#include "TempLat/lattice/algebra/operators/operators.h"
#include "CosmoInterface/definitions/gaugederivatives.h"
#include "TempLat/lattice/algebra/gaugealgebra/forwardcovariantderivative.h"
#include "TempLat/lattice/algebra/su2algebra/su2multiply.h"
#include "TempLat/lattice/algebra/gaugealgebra/fieldstrength.h"
#include "TempLat/lattice/algebra/gaugealgebra/plaquette.h"
#include "TempLat/util/rangeiteration/for_in_range.h"
#include "TempLat/util/rangeiteration/sum_in_range.h"
#include "TempLat/lattice/algebra/operators/power.h"
#include "TempLat/lattice/algebra/spatialderivatives/normgradientsquare.h"
#include "CosmoInterface/runparameters.h"
namespace TempLat {


    /** \brief A class which stores expressions of field functionals.
     *
     * 
     **/

    class FieldFunctionals {
    public:
        /* Put public methods here. These should change very little over time. */

        FieldFunctionals() = delete;
        
        // By "field functional" we refer to any function of the field variables such as quadratic forms, e.g. <phi^2>, and others.
                
        // The following functions compute the field functionals that appear in the total energy density,
        // but without the corresponding multiplying factors. For example, here we compute <Pi^2> for a scalar singlet, 
        // while in energies.h we multiply it by 1/2*a^(-6) to get the corresponding kinetic energy.

		// --> Scalar singlet:
		
        template<class Model, int I>  // <Grad[f]^2>
        static inline auto fieldConfig(Model& model, Tag<I> i)
        {
            return model.fldS(i);
        }

        //--> Squared sum of scalars

        /////////OnlyforZ3, field has to be normalized properly using the vacuum
        ////define M_PI in a clean way
        template<class Model, typename T>
        static inline auto Z3vacuumPhase(Model& model, T lSide, int N)
        {
            auto phi1 = model.fldS(0_c);  // real part
            auto phi2 = model.fldS(1_c);  // imaginary part
             //it will probably not work
            // arg returns phase in [0, 2π]
            auto theta = atan2(phi2, phi1);  // returns [-π, π]

            // Shift to [0, 2π] and divide into thirds
            auto index =  floor((theta + 3.1415) / (2.0 * 3.1415 / 3.0));
            //Shift to compare
            auto index_shift_x = shift<1>(index);
            auto index_shift_y = shift<2>(index);   
            auto index_shift_z = shift<3>(index);
            //
            //Type12
            auto is12z = (index == 1) * (index_shift_z == 2);
            auto is12y = (index == 1) * (index_shift_y == 2);
            auto is12x = (index == 1) * (index_shift_x == 2);
            auto is21z = (index == 2) * (index_shift_z == 1);
            auto is21y = (index == 2) * (index_shift_y == 1);
            auto is21x = (index == 2) * (index_shift_x == 1);
            //Type02
            auto is02z = (index == 0) * (index_shift_z == 2);
            auto is02y = (index == 0) * (index_shift_y == 2);
            auto is02x = (index == 0) * (index_shift_x == 2);
            auto is20z = (index == 2) * (index_shift_z == 0);
            auto is20y = (index == 2) * (index_shift_y == 0);
            auto is20x = (index == 2) * (index_shift_x == 0);
            //Type10
            auto is10z = (index == 1) * (index_shift_z == 0);
            auto is10y = (index == 1) * (index_shift_y == 0);
            auto is10x = (index == 1) * (index_shift_x == 0);
            auto is01z = (index == 0) * (index_shift_z == 1);
            auto is01y = (index == 0) * (index_shift_y == 1);
            auto is01x = (index == 0) * (index_shift_x == 1);
            auto numnormscal = sqrt(  pow<2>(forwDiff(theta, Tag<1>()))
                            + pow<2>(forwDiff(theta, Tag<2>()))
                            + pow<2>(forwDiff(theta, Tag<3>())));
            auto dennormscal = sqrt(pow<2>(forwDiff(theta, Tag<1>())))
                       + sqrt(pow<2>(forwDiff(theta, Tag<2>())))
                       + sqrt(pow<2>(forwDiff(theta, Tag<3>())));
            //just remain to implement the normalization with the derivatives
            auto deltax = lSide/N; //will it make sense?
            return model.aI*pow<2>(deltax)*numnormscal/dennormscal*(is12z + is12y + is12x + is21z + is21y + is21x + is02z + is02y + is02x + is20z + is20y + is20x + is10z + is10y + is10x + is01z + is01y + is01x)/pow<3>(lSide); //This should be final version!
        }
        
        // -------------------------------------------------------------------------
        // ZNvacuumPhase: domain wall area for one wall type in a ZN model.
        //
        // Template parameters:
        //   Nwall    – number of degenerate vacua
        //   WallType – 1 = adjacent vacua, 2 = next-to-adjacent, …, floor(Nwall/2).
        //              For even Nwall, type Nwall/2 connects antipodal vacua.
        //
        // Call this once per wall type from model::wallAreaTerm(Tag<WallType-1>).
        // EnergiesMeasurer loops over all types and writes each as a separate column.
        //
        // Gradient correction (PRS formula):
        //   Uses forwDiff of phi1, phi2 (= ∂_i phi, units field/length).
        //   numnorm/dennorm is dimensionless (units cancel), Δx² provides area units.
        //   This avoids the atan2 branch-cut discontinuity at θ=±π.
        // -------------------------------------------------------------------------
        template<int Nwall, int WallType, class Model, typename T>
        static inline auto ZNvacuumPhase(Model& model, T lSide, int N)
        {
            static_assert(WallType >= 1 && WallType * 2 <= Nwall,
                          "WallType must satisfy 1 <= WallType <= floor(Nwall/2)");

            constexpr double PI = 3.14159265358979323846;

            auto phi1 = model.fldS(0_c);
            auto phi2 = model.fldS(1_c);

            auto theta = atan2(phi2, phi1);  // (-π, π]

            // Vacuum index in {0,…,Nwall-1}. Offset π/(2N) centres sectors on vacua θ=2πk/N.
            // mod Nwall handles the atan2 branch cut cleanly for all N.
            auto raw_index = floor(T(Nwall) * (theta + T(PI) + T(PI / Nwall)) / T(2.0 * PI));
            auto index     = raw_index - T(Nwall) * floor(raw_index / T(Nwall));

            auto dx = index - shift<1>(index);
            auto dy = index - shift<2>(index);
            auto dz = index - shift<3>(index);

            // PRS gradient correction: numnorm/dennorm is dimensionless (1/length cancels).
            auto numnorm = sqrt(pow<2>(forwDiff(phi1, Tag<1>())) + pow<2>(forwDiff(phi1, Tag<2>())) + pow<2>(forwDiff(phi1, Tag<3>()))
                              + pow<2>(forwDiff(phi2, Tag<1>())) + pow<2>(forwDiff(phi2, Tag<2>())) + pow<2>(forwDiff(phi2, Tag<3>())));
            auto dennorm = sqrt(pow<2>(forwDiff(phi1, Tag<1>())) + pow<2>(forwDiff(phi2, Tag<1>())))
                         + sqrt(pow<2>(forwDiff(phi1, Tag<2>())) + pow<2>(forwDiff(phi2, Tag<2>())))
                         + sqrt(pow<2>(forwDiff(phi1, Tag<3>())) + pow<2>(forwDiff(phi2, Tag<3>())));

            auto deltax    = lSide / T(N);
            auto area_elem = model.aI * pow<2>(deltax) * numnorm / (dennorm + T(1e-30)) / pow<3>(lSide);

            // Crossing indicators: raw index diff is ±WallType (short arc) or ±(Nwall-WallType) (long arc).
            // For the antipodal case WallType = Nwall/2, both arcs have equal length — use only ±WallType.
            if constexpr (WallType * 2 == Nwall) {
                auto crossings = (dx == T( WallType)) + (dx == T(-WallType))
                               + (dy == T( WallType)) + (dy == T(-WallType))
                               + (dz == T( WallType)) + (dz == T(-WallType));
                return area_elem * crossings;
            } else {
                auto crossings = (dx == T( WallType))         + (dx == T(-WallType))
                               + (dx == T( Nwall - WallType)) + (dx == T(WallType - Nwall))
                               + (dy == T( WallType))         + (dy == T(-WallType))
                               + (dy == T( Nwall - WallType)) + (dy == T(WallType - Nwall))
                               + (dz == T( WallType))         + (dz == T(-WallType))
                               + (dz == T( Nwall - WallType)) + (dz == T(WallType - Nwall));
                return area_elem * crossings;
            }
        }

        ///to debug and then good!


        ///Use shift!!!


        //Now you have to define a method which compares two neighboring sites and see if they are in the same vacuum or not.
        //This can be done in the energies.h file when computing the potential energy density.

        template<class Model, int I>  // <Grad[f]^2>
        static inline auto grad2S(Model& model, Tag<I> i)
        {
            return Grad2<Model::NDim>(model.fldS(i));
        }

        template<class Model, int I>  // <pi^2>
        static inline auto pi2S(Model& model, Tag<I> i)
        {
            return pow<2>(model.piS(i));
        }

		// --> Complex scalar:
		
        template <class Model, int I>  // <D_i[f]^2> (sum over i)
        static inline auto grad2CS(Model& model, Tag<I> i)
        {
            return Total(j, 1, Model::NDim, norm2( GaugeDerivatives::forwardCovGradientCS(model,i,j)));
        }

        template <class Model, int I>  // <pi^2>
        static inline auto pi2CS(Model& model, Tag<I> i)
        {
            return norm2(model.piCS(i));
        }

		// --> SU2 doublet:
		
        template <class Model, int I>   // <D_i[f]^2> (sum over i)
        static inline auto grad2SU2Doublet(Model& model, Tag<I> i)
        {

            return Total(j, 1, Model::NDim, norm2( GaugeDerivatives::forwardCovGradientSU2Doublet(model,i,j)));
        }

        template <class Model, int I>   // <pi^2>
        static inline auto pi2SU2Doublet(Model& model,Tag<I> i)
        {
            return norm2(model.piSU2Doublet(i));
        }

		// --> U1 gauge sector:

        template <class Model, int A>
        static inline auto B2U1(Model& model, Tag<A> a)  // In 3D, returns F_{21}^2 + F_{31}^2 + F_{32}^2 (necessary to compute the magnetic energy) 
        {
            return   Total(i,1,Model::NDim,
                                        Total(j,1,Model::NDim,
                                              IfElse(IsLess(j,i),
                                                     pow<2>(fieldStrength(model.fldU1(a),i,j)), ZeroType());
                                        ));
        }
        
        template <class Model, int A>    // <pi^2>
        static inline auto pi2U1(Model& model, Tag<A> a)
        {
            return  Total(i,1,Model::NDim,
                                       pow<2>(model.piU1(a)(i))
            );
        }
        
        // --> SU2 gauge sector:

        template <class Model,int A>
        static inline auto B2SU2(Model& model, Tag<A> a)
        {
            return 4.0 / (pow<4>(model.dx) * pow<2>(model.gQ_SU2DblSU2(0_c, a)))*  Total(i,1,Model::NDim,
                                               Total(j,1,Model::NDim,
                                                       IfElse(IsLess(j,i),
                                                        2.0 - trace(plaq(model.fldSU2(a),i,j)), //if
                                                       ZeroType() //else
                                                            );
                                                     )
                                               );
        }
        template <class Model, int A> // <pi^2>
        static inline auto pi2SU2(Model& model, Tag<A> a)
        {

            return Total(i,1,Model::NDim,
                         Total(b,1,3,
                                    pow<2>(model.piSU2(a)(i).SU2LieAlgebraGet(b))
                                )
                     );
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
#include "CosmoInterface/definitions/fieldfunctionals_test.h"
#endif


#endif
