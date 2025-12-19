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
            auto numnormscal = sqrt(pow<2>(forwDiff(phi1, Tag<1>())) + pow<2>(forwDiff(phi1, Tag<2>())) + pow<2>(forwDiff(phi1, Tag<3>())) + pow<2>(forwDiff(phi2, Tag<1>())) + pow<2>(forwDiff(phi2, Tag<2>())) + pow<2>(forwDiff(phi2, Tag<3>())));
            auto dennormscal = sqrt(pow<2>(forwDiff(phi1, Tag<1>())) + pow<2>(forwDiff(phi2, Tag<1>()))) + sqrt(pow<2>(forwDiff(phi1, Tag<2>())) + pow<2>(forwDiff(phi2, Tag<2>()))) + sqrt(pow<2>(forwDiff(phi1, Tag<3>())) + pow<2>(forwDiff(phi2, Tag<3>())));
            //just remain to implement the normalization with the derivatives
            auto deltax = lSide/N; //will it make sense?
            return model.aI*pow<2>(deltax)*numnormscal/dennormscal*(is12z + is12y + is12x + is21z + is21y + is21x + is02z + is02y + is02x + is20z + is20y + is20x + is10z + is10y + is10x + is01z + is01y + is01x)/pow<3>(lSide); //This should be final version!
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
