#ifndef DWZ4_H
#define DWZ4_H

#include "CosmoInterface/cosmointerface.h"

namespace TempLat
{
    /////////
    // Z4 domain wall model: complex scalar field Φ = (h + ia)/√2 with
    //
    //   V = -(μ²/2)(h²+a²) + (λ₁/4)(h²+a²)² − (λ₂/2)(h⁴ − 6h²a² + a⁴) + V₀
    //
    // The Z4-breaking term −(λ₂/2)(h⁴−6h²a²+a⁴) = −λ₂ Re(Φ⁴) creates 4 degenerate
    // vacua at θ = 0, π/2, π, −π/2  (i.e. h = ±v, a = 0  and  h = 0, a = ±v)
    // with vacuum field value  v = μ/√(λ₁ − 2μλ₂)  (requires λ₁ > 2μλ₂ > 0).
    //
    // Two wall types exist:
    //   Type 1 (adjacent, WallType=1): connects vacua differing by one Z4 step (θ→θ+π/2).
    //   Type 2 (diagonal, WallType=2): connects antipodal vacua (θ→θ+π), higher tension,
    //          expected to be rare and to split into two type-1 walls.
    //
    // Output columns: Scal_type_1 (adjacent area parameter), Scal_type_2 (diagonal).
    /////////

    struct ModelPars : public TempLat::DefaultModelPars {
        static constexpr size_t NScalars  = 2;
        static constexpr size_t NPotTerms = 3;
        // Z4: floor(4/2) = 2 distinct wall types.
        static constexpr size_t NWallTypes = 2;
    };

  #define MODELNAME DWZ4

  template<class R>
  using Model = MakeModel(R, ModelPars);

  class MODELNAME : public Model<MODELNAME>
  {
  private:
    double lambda1, lambda2, mu;

  public:

    MODELNAME(ParameterParser& parser, RunParameters<double>& runPar,
              std::shared_ptr<MemoryToolBox> toolBox)
    : Model<MODELNAME>(parser, runPar.getLatParams(), toolBox, runPar.dt, STRINGIFY(MODELLABEL))
    {
        lambda1 = parser.get<double>("lambda1");
        lambda2 = parser.get<double>("lambda2");
        mu      = parser.get<double>("mu");

        fldS0 = parser.get<double, 2>("initial_amplitudes");
        piS0  = parser.get<double, 2>("initial_momenta", {0, 0});

        // Vacuum: h = v = μ/√(λ₁−2μλ₂), a = 0.  Requires λ₁ > 2μλ₂.
        alpha     = 1;
        fStar     = mu / sqrt(lambda1 - 2.0 * lambda2 * mu);
        omegaStar = mu;

        setInitialPotentialAndMassesFromPotential();
    }

    /////////
    // Potential terms
    /////////

    // Term 0: mass term  −(μ²/2)(h²+a²)  plus constant V₀ = μ⁴/[4(λ₁−2μλ₂)] that
    //         zeroes the vacuum energy at h=v, a=0.
    auto potentialTerms(Tag<0>)
    {
        return -(0.5) * mu * mu * (pow<2>(fldS(0_c)) + pow<2>(fldS(1_c)))
               + pow(mu, 4) / (4.0 * (lambda1 - 2.0 * lambda2* mu));
    }

    // Term 1: quartic  (λ₁/4)(h²+a²)²
    auto potentialTerms(Tag<1>)
    {
        return (0.25) * lambda1 * pow<2>(pow<2>(fldS(0_c)) + pow<2>(fldS(1_c)));
    }

    // Term 2: Z4-breaking  −(λ₂/2)(h⁴ − 6h²a² + a⁴)
    auto potentialTerms(Tag<2>)
    {
        return -(0.5) * lambda2* mu * (pow<4>(fldS(0_c))
                                   - 6.0 * pow<2>(fldS(0_c)) * pow<2>(fldS(1_c))
                                   + pow<4>(fldS(1_c)));
    }

    /////////
    // First derivatives of V (equations of motion)
    /////////

    // dV/dh = −μ²h + λ₁h(h²+a²) − 2λ₂h(h² − 3a²)
    auto potDeriv(Tag<0>)
    {
        return -mu * mu * fldS(0_c)
               + lambda1 * fldS(0_c) * (pow<2>(fldS(0_c)) + pow<2>(fldS(1_c)))
               - 2.0 * lambda2 * mu * fldS(0_c) * (pow<2>(fldS(0_c)) - 3.0 * pow<2>(fldS(1_c)));
    }

    // dV/da = −μ²a + λ₁a(h²+a²) + 2λ₂a(3h² − a²)
    auto potDeriv(Tag<1>)
    {
        return -mu * mu * fldS(1_c)
               + lambda1 * fldS(1_c) * (pow<2>(fldS(0_c)) + pow<2>(fldS(1_c)))
               + 2.0 * lambda2 * mu * fldS(1_c) * (3.0 * pow<2>(fldS(0_c)) - pow<2>(fldS(1_c)));
    }

    /////////
    // Second derivatives of V (used for mass initialisation)
    /////////

    // d²V/dh² = −μ² + (3λ₁−6λ₂)h² + (λ₁+6λ₂)a²
    auto potDeriv2(Tag<0>)
    {
        return -mu * mu
               + (3.0 * lambda1 - 6.0 * lambda2 * mu) * pow<2>(fldS(0_c))
               + (lambda1 + 6.0 * lambda2 * mu)        * pow<2>(fldS(1_c));
    }

    // d²V/da² = −μ² + (λ₁+6λ₂)h² + (3λ₁−6λ₂)a²
    auto potDeriv2(Tag<1>)
    {
        return -mu * mu
               + (lambda1 + 6.0 * lambda2 *mu)        * pow<2>(fldS(0_c))
               + (3.0 * lambda1 - 6.0 * lambda2 * mu)  * pow<2>(fldS(1_c));
    }

    /////////
    // Wall area terms — one per wall type, 0-based index.
    //   Tag<0> → WallType 1 (adjacent)
    //   Tag<1> → WallType 2 (diagonal / antipodal)
    /////////
    template<int WallIdx>
    auto wallAreaTerm(Tag<WallIdx>, double lSide_in, int N_in)
    {
        return FieldFunctionals::ZNvacuumPhase<4, WallIdx + 1>(*this, lSide_in, N_in);
    }

    };
}

#endif // DWZ4_H
