#ifndef DWZ4_H
#define DWZ4_H

#include "CosmoInterface/cosmointerface.h"
#include "TempLat/lattice/measuringtools/scaling.h"
#include <vector>
#include <string>
#include <utility>

namespace TempLat
{
    /////////
    // Z4 domain wall model: complex scalar field Φ = (h + ia)/√2 with
    //
    //   V = -μ²|Φ|² + λ₁|Φ|⁴ − λ₂(Φ⁴ + Φ*⁴)
    //     = -(μ²/2)(h²+a²) + (λ₁/4)(h²+a²)² − (λ₂/2)(h⁴ − 6h²a² + a⁴) + V₀
    //
    // The Z4-breaking term −(λ₂/2)(h⁴−6h²a²+a⁴) = −λ₂ Re(Φ⁴) creates 4 degenerate
    // vacua at θ = 0, π/2, π, −π/2  (i.e. h = ±v, a = 0  and  h = 0, a = ±v)
    // with vacuum field value  v = μ/√(λ₁ − 2λ₂)  (requires λ₁ > 2λ₂ > 0).
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

        // Vacuum: h = v = μ/√(λ₁−2λ₂), a = 0.  Requires λ₁ > 2λ₂.
        alpha     = 1;
        fStar     = mu / sqrt(lambda1 - 2.0 * lambda2);
        omegaStar = mu;

        setInitialPotentialAndMassesFromPotential();
    }

    /////////
    // Potential terms
    /////////

    // Term 0: mass term  −(μ²/2)(h²+a²)  plus constant V₀ = μ⁴/[4(λ₁−2λ₂)] that
    //         zeroes the vacuum energy at h=v, a=0.
    auto potentialTerms(Tag<0>)
    {
        return -(0.5) * mu * mu * (pow<2>(fldS(0_c)) + pow<2>(fldS(1_c)))
               + pow(mu, 4) / (4.0 * (lambda1 - 2.0 * lambda2));
    }

    // Term 1: quartic  (λ₁/4)(h²+a²)²
    auto potentialTerms(Tag<1>)
    {
        return (0.25) * lambda1 * pow<2>(pow<2>(fldS(0_c)) + pow<2>(fldS(1_c)));
    }

    // Term 2: Z4-breaking  −(λ₂/2)(h⁴ − 6h²a² + a⁴)
    auto potentialTerms(Tag<2>)
    {
        return -(0.5) * lambda2 * (pow<4>(fldS(0_c))
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
               - 2.0 * lambda2 * fldS(0_c) * (pow<2>(fldS(0_c)) - 3.0 * pow<2>(fldS(1_c)));
    }

    // dV/da = −μ²a + λ₁a(h²+a²) + 2λ₂a(3h² − a²)
    auto potDeriv(Tag<1>)
    {
        return -mu * mu * fldS(1_c)
               + lambda1 * fldS(1_c) * (pow<2>(fldS(0_c)) + pow<2>(fldS(1_c)))
               + 2.0 * lambda2 * fldS(1_c) * (3.0 * pow<2>(fldS(0_c)) - pow<2>(fldS(1_c)));
    }

    /////////
    // Second derivatives of V (used for mass initialisation)
    /////////

    // d²V/dh² = −μ² + (3λ₁−6λ₂)h² + (λ₁+6λ₂)a²
    auto potDeriv2(Tag<0>)
    {
        return -mu * mu
               + (3.0 * lambda1 - 6.0 * lambda2) * pow<2>(fldS(0_c))
               + (lambda1 + 6.0 * lambda2)        * pow<2>(fldS(1_c));
    }

    // d²V/da² = −μ² + (λ₁+6λ₂)h² + (3λ₁−6λ₂)a²
    auto potDeriv2(Tag<1>)
    {
        return -mu * mu
               + (lambda1 + 6.0 * lambda2)        * pow<2>(fldS(0_c))
               + (3.0 * lambda1 - 6.0 * lambda2)  * pow<2>(fldS(1_c));
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

    /////////
    // Junction (theta-winding string) + wall-velocity diagnostics.
    // Written to a separate "wall_diagnostics" file on the INFREQUENT schedule
    // (the winding counter is transcendental-heavy).  EnergiesMeasurer detects
    // these methods via SFINAE and skips models that do not define them.
    //
    //   Junction:  Jstr_len (a*dx/L^3 length density, |w|>=1), Jstr_cnt (raw
    //              pierced faces), Jstr_signed (net winding ~0 check),
    //              Jstr_w2 (|w|>=2 faces), Jpi_edge (antipodal-edge resolution
    //              flag), Jcore_cnt (|Phi| < 0.3 v core sites).
    //   Velocity:  v^2 gamma^2 = sum(mask*KE)/sum(mask*V) over wall sites.  We
    //              emit the three raw sums {KE,V,N} per selector so the ratio
    //              (ratio-of-sums vs average-of-ratios) is formed in analysis:
    //                A = potential cutoff V > alpha*Vmax (alpha=0.2,0.4,0.6),
    //                B = ZNvacuumPhase wall-site mask (adjacent / non-adj / all).
    /////////

    // Per-site physical kinetic-energy density of the two scalars (0.5 a^-6 pi^2).
    auto wallKineticDensity()
    {
        return 0.5 * pow<-6>(aI) * (pow<2>(piS(0_c)) + pow<2>(piS(1_c)));
    }

    // Per-site (vacuum-subtracted) potential-energy density.
    auto wallPotentialDensity()
    {
        return potentialTerms(Tag<0>()) + potentialTerms(Tag<1>()) + potentialTerms(Tag<2>());
    }

    std::vector<std::string> wallDiagnosticsHeaders() const
    {
        return { "Jstr_len", "Jstr_cnt", "Jstr_signed", "Jstr_w2", "Jpi_edge", "Jcore_cnt",
                 "velA02_KE", "velA02_V", "velA02_N",
                 "velA04_KE", "velA04_V", "velA04_N",
                 "velA06_KE", "velA06_V", "velA06_N",
                 "velBadj_KE", "velBadj_V", "velBadj_N",
                 "velBnon_KE", "velBnon_V", "velBnon_N",
                 "velBall_KE", "velBall_V", "velBall_N" };
    }

    std::vector<double> wallDiagnosticsValues(double lSide_in, int N_in)
    {
        using FF = FieldFunctionals;
        std::vector<double> out;
        out.reserve(24);

        // --- junction (theta-winding) string ---
        out.push_back(scale(FF::ZNstringDensity(*this, lSide_in, N_in)));
        out.push_back(scale(FF::ZNstringCount  (*this, lSide_in, N_in)));
        out.push_back(scale(FF::ZNstringSigned (*this, lSide_in, N_in)));
        out.push_back(scale(FF::ZNstringW2     (*this, lSide_in, N_in)));
        out.push_back(scale(FF::ZNantipodalEdges<4>(*this, lSide_in, N_in)));
        const double v2 = mu * mu / (lambda1 - 2.0 * lambda2);          // vacuum |fld|^2
        out.push_back(scale(FF::fieldCoreIndicator(*this, 0.09 * v2))); // eps = 0.3

        // --- wall velocity:  v^2 gamma^2 = sum(mask*KE)/sum(mask*V) ---
        // adjacent-wall barrier height (vacuum-subtracted); the lower barrier, so
        // alpha*Vmax catches both wall types.
        const double Vmax = pow(mu, 4) * lambda2 / (lambda1 * lambda1 - 4.0 * lambda2 * lambda2);
        auto KE = wallKineticDensity();
        auto Vd = wallPotentialDensity();

        // selector A: potential cutoff (all walls)
        for (double alpha : {0.2, 0.4, 0.6}) {
            auto m = heaviside(Vd - alpha * Vmax);
            out.push_back(scale(m * KE));
            out.push_back(scale(m * Vd));
            out.push_back(scale(m));
        }

        // selector B: ZNvacuumPhase wall-site masks (adjacent / non-adjacent / union)
        auto mAdj = FF::ZNwallSiteMask<4, 1>(*this, lSide_in, N_in);
        auto mNon = FF::ZNwallSiteMask<4, 2>(*this, lSide_in, N_in);
        auto mAll = heaviside(mAdj + mNon - 0.5);
        out.push_back(scale(mAdj * KE)); out.push_back(scale(mAdj * Vd)); out.push_back(scale(mAdj));
        out.push_back(scale(mNon * KE)); out.push_back(scale(mNon * Vd)); out.push_back(scale(mNon));
        out.push_back(scale(mAll * KE)); out.push_back(scale(mAll * Vd)); out.push_back(scale(mAll));

        return out;
    }

    };
}

#endif // DWZ4_H
