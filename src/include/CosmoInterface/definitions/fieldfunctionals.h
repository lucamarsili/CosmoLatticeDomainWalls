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

            // Vacuum index in {0,…,Nwall-1}, centred on vacua θ=2πk/Nwall.
            // mod Nwall handles the atan2 branch cut cleanly for all N.
            // Parity-dependent sector offset: half-bin (PI/Nwall) centres vacua only for EVEN
            // Nwall; for ODD Nwall (e.g. Z3) it puts vacua exactly ON bin boundaries -> ~50%
            // spurious bulk-vacuum crossings (flicker).  Odd Nwall needs offset 0.  (constexpr
            // -> folded at compile time.)
            constexpr double sectorOffset = (Nwall % 2 == 0) ? (PI / Nwall) : 0.0;
            auto raw_index = floor(T(Nwall) * (theta + T(PI) + T(sectorOffset)) / T(2.0 * PI));
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

        // -------------------------------------------------------------------------
        // Z2vacuumSign: domain-wall area for a single REAL scalar with Z2 vacua +-v.
        //
        // A real scalar has no phase => no winding => no global-string cores, hence
        // NO junctions: this is the junction-free counterpart of ZNvacuumPhase, used
        // as the clean null-test for the PRS-vs-physical area gap.  The vacuum label
        // is sign(phi) in {0,1}; a wall face is a neighbour sign flip.  The gradient/
        // orientation correction (numnorm/dennorm, Manhattan->Euclidean) and the
        // a*dx^2/L^3 area element are IDENTICAL to ZNvacuumPhase (single-field form).
        // Algebraically equals ZNvacuumPhase<2,1> evaluated on an embedded (phi, 0).
        // -------------------------------------------------------------------------
        template<class Model, typename T>
        static inline auto Z2vacuumSign(Model& model, T lSide, int N)
        {
            auto phi = model.fldS(0_c);
            auto s   = heaviside(phi);          // vacuum label: 0 (phi<0) or 1 (phi>0)

            auto dx = s - shift<1>(s);
            auto dy = s - shift<2>(s);
            auto dz = s - shift<3>(s);

            // PRS gradient/orientation correction (single real field):
            //   numnorm = |grad phi| (Euclidean), dennorm = sum_i |d_i phi| (Manhattan).
            auto numnorm = sqrt(pow<2>(forwDiff(phi, Tag<1>()))
                              + pow<2>(forwDiff(phi, Tag<2>()))
                              + pow<2>(forwDiff(phi, Tag<3>())));
            auto dennorm = sqrt(pow<2>(forwDiff(phi, Tag<1>())))
                         + sqrt(pow<2>(forwDiff(phi, Tag<2>())))
                         + sqrt(pow<2>(forwDiff(phi, Tag<3>())));

            auto deltax    = lSide / T(N);
            auto area_elem = model.aI * pow<2>(deltax) * numnorm / (dennorm + T(1e-30)) / pow<3>(lSide);

            // Wall face = neighbour sign flip (|index diff| = 1).
            auto crossings = (dx == T( 1)) + (dx == T(-1))
                           + (dy == T( 1)) + (dy == T(-1))
                           + (dz == T( 1)) + (dz == T(-1));
            return area_elem * crossings;
        }

        // =====================================================================
        // Junction / global-string diagnostics for ZN domain-wall networks.
        //
        // A Z_N wall junction is the line around which the phase theta winds by
        // 2*pi (a |Phi|->0 global-string core).  We detect it with the standard
        // geodesic plaquette-winding counter (Kibble / Vachaspati-Vilenkin):
        // around each elementary lattice face, sum the four shortest-arc phase
        // differences; the sum is ~0 off a string and ~+-2*pi through a core.
        //
        // The geodesic rule  d -> atan2(sin d, cos d) in (-pi,pi] is exact when
        // every edge is sub-pi (resolved core); it is degenerate for an exactly
        // antipodal (pi) edge -- an under-resolved double wall -- which is what
        // ZNantipodalEdges quantifies as a built-in resolution self-diagnostic.
        // =====================================================================

        // Geodesic winding sum around the elementary (Mu,Nu) plaquette: ~0 or ~+-2pi.
        // Shortest-arc wrap of each edge via  wrap(d) = d - 2pi*floor(d/2pi + 1/2),
        // mapping d into (-pi,pi].  (floor-based, so no dependence on transcendental
        // operators; the four raw diffs telescope to 0, leaving 2pi*winding.)
        template<int Mu, int Nu, class TH>
        static inline auto plaqWindingSum(TH th)
        {
            constexpr double PI  = 3.14159265358979323846;
            constexpr double TPI = 2.0 * PI;
            auto d1 = shift<Mu>(th) - th;                         // 00   -> Mu0
            auto d2 = shift<Mu>(shift<Nu>(th)) - shift<Mu>(th);   // Mu0  -> MuNu
            auto d3 = shift<Nu>(th) - shift<Mu>(shift<Nu>(th));   // MuNu -> 0Nu
            auto d4 = th - shift<Nu>(th);                         // 0Nu  -> 00
            return (d1 - TPI * floor(d1 / TPI + 0.5))
                 + (d2 - TPI * floor(d2 / TPI + 0.5))
                 + (d3 - TPI * floor(d3 / TPI + 0.5))
                 + (d4 - TPI * floor(d4 / TPI + 0.5));
        }

        // Comoving string-length density (faces with |winding| >= 1), scaled a*dx/L^3.
        template<class Model, typename T>
        static inline auto ZNstringDensity(Model& model, T lSide, int N)
        {
            constexpr double PI = 3.14159265358979323846;
            auto th = atan2(model.fldS(1_c), model.fldS(0_c));
            auto pierced = heaviside(abs(plaqWindingSum<1,2>(th)) - PI)
                         + heaviside(abs(plaqWindingSum<2,3>(th)) - PI)
                         + heaviside(abs(plaqWindingSum<3,1>(th)) - PI);
            auto deltax = lSide / T(N);
            return model.aI * deltax * pierced / pow<3>(lSide);
        }

        // Raw pierced-face count (|winding| >= 1), no normalization.
        template<class Model, typename T>
        static inline auto ZNstringCount(Model& model, T lSide, int N)
        {
            constexpr double PI = 3.14159265358979323846;
            auto th = atan2(model.fldS(1_c), model.fldS(0_c));
            return heaviside(abs(plaqWindingSum<1,2>(th)) - PI)
                 + heaviside(abs(plaqWindingSum<2,3>(th)) - PI)
                 + heaviside(abs(plaqWindingSum<3,1>(th)) - PI);
        }

        // Net signed winding per face, summed over orientations (global check ~ 0).
        // Symmetric +-1 form: +1 for W>=+pi, -1 for W<=-pi, 0 otherwise.  (Avoids the
        // round-half-up bias of nint at the W=+-pi noise boundary; real windings have
        // W~+-2pi and are counted with the correct sign either way.)
        template<class Model, typename T>
        static inline auto ZNstringSigned(Model& model, T lSide, int N)
        {
            constexpr double PI = 3.14159265358979323846;
            auto th  = atan2(model.fldS(1_c), model.fldS(0_c));
            auto w12 = plaqWindingSum<1,2>(th);
            auto w23 = plaqWindingSum<2,3>(th);
            auto w31 = plaqWindingSum<3,1>(th);
            return (heaviside(w12 - PI) - heaviside(-w12 - PI))
                 + (heaviside(w23 - PI) - heaviside(-w23 - PI))
                 + (heaviside(w31 - PI) - heaviside(-w31 - PI));
        }

        // High-winding faces (|winding| >= 2): rare; flags artifacts / multi-strings.
        template<class Model, typename T>
        static inline auto ZNstringW2(Model& model, T lSide, int N)
        {
            constexpr double PI = 3.14159265358979323846;
            auto th = atan2(model.fldS(1_c), model.fldS(0_c));
            return heaviside(abs(plaqWindingSum<1,2>(th)) - 3.0 * PI)
                 + heaviside(abs(plaqWindingSum<2,3>(th)) - 3.0 * PI)
                 + heaviside(abs(plaqWindingSum<3,1>(th)) - 3.0 * PI);
        }

        // Resolution self-diagnostic: count of antipodal-adjacent edges (vacuum
        // index differs by Nwall/2 between neighbours) -- exactly the cases where
        // the winding rule is ambiguous.  scale(...) -> total antipodal edges.
        template<int Nwall, class Model, typename T>
        static inline auto ZNantipodalEdges(Model& model, T lSide, int N)
        {
            constexpr double PI = 3.14159265358979323846;
            auto theta     = atan2(model.fldS(1_c), model.fldS(0_c));
            // Parity-dependent sector offset: half-bin (PI/Nwall) centres vacua only for EVEN
            // Nwall; for ODD Nwall (e.g. Z3) it puts vacua exactly ON bin boundaries -> ~50%
            // spurious bulk-vacuum crossings (flicker).  Odd Nwall needs offset 0.  (constexpr
            // -> folded at compile time.)
            constexpr double sectorOffset = (Nwall % 2 == 0) ? (PI / Nwall) : 0.0;
            auto raw_index = floor(T(Nwall) * (theta + T(PI) + T(sectorOffset)) / T(2.0 * PI));
            auto index     = raw_index - T(Nwall) * floor(raw_index / T(Nwall));
            auto dx = index - shift<1>(index);
            auto dy = index - shift<2>(index);
            auto dz = index - shift<3>(index);
            constexpr int half = Nwall / 2;
            return (dx == T( half)) + (dx == T(-half))
                 + (dy == T( half)) + (dy == T(-half))
                 + (dz == T( half)) + (dz == T(-half));
        }

        // |Phi|^2 = (phi1^2+phi2^2) core indicator: 1 where magnitude^2 < thr2.
        // Cross-checks the winding count -- true junction cores have |Phi|->0.
        template<class Model, typename T>
        static inline auto fieldCoreIndicator(Model& model, T thr2)
        {
            auto r2 = pow<2>(model.fldS(0_c)) + pow<2>(model.fldS(1_c));
            return heaviside(thr2 - r2);
        }

        // On-wall site mask for one wall type (>=1 ZN-sector crossing on its
        // forward faces).  Used as the velocity wall-selector "B".
        template<int Nwall, int WallType, class Model, typename T>
        static inline auto ZNwallSiteMask(Model& model, T lSide, int N)
        {
            constexpr double PI = 3.14159265358979323846;
            auto theta     = atan2(model.fldS(1_c), model.fldS(0_c));
            // Parity-dependent sector offset: half-bin (PI/Nwall) centres vacua only for EVEN
            // Nwall; for ODD Nwall (e.g. Z3) it puts vacua exactly ON bin boundaries -> ~50%
            // spurious bulk-vacuum crossings (flicker).  Odd Nwall needs offset 0.  (constexpr
            // -> folded at compile time.)
            constexpr double sectorOffset = (Nwall % 2 == 0) ? (PI / Nwall) : 0.0;
            auto raw_index = floor(T(Nwall) * (theta + T(PI) + T(sectorOffset)) / T(2.0 * PI));
            auto index     = raw_index - T(Nwall) * floor(raw_index / T(Nwall));
            auto dx = index - shift<1>(index);
            auto dy = index - shift<2>(index);
            auto dz = index - shift<3>(index);
            if constexpr (WallType * 2 == Nwall) {
                auto crossings = (dx == T( WallType)) + (dx == T(-WallType))
                               + (dy == T( WallType)) + (dy == T(-WallType))
                               + (dz == T( WallType)) + (dz == T(-WallType));
                return heaviside(crossings - 0.5);
            } else {
                auto crossings = (dx == T( WallType))         + (dx == T(-WallType))
                               + (dx == T( Nwall - WallType)) + (dx == T(WallType - Nwall))
                               + (dy == T( WallType))         + (dy == T(-WallType))
                               + (dy == T( Nwall - WallType)) + (dy == T(WallType - Nwall))
                               + (dz == T( WallType))         + (dz == T(-WallType))
                               + (dz == T( Nwall - WallType)) + (dz == T(WallType - Nwall));
                return heaviside(crossings - 0.5);
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
