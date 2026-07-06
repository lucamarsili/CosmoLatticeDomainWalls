#ifndef DWZ2_H
#define DWZ2_H

#include "CosmoInterface/cosmointerface.h"
#include "TempLat/lattice/measuringtools/scaling.h"
#include <vector>
#include <string>

namespace TempLat
{
    /////////
    // Z2 domain wall model: a single REAL scalar phi with
    //
    //   V = -(mu^2/2) phi^2 + (Lambda/4) phi^4 + V0 = (mu^2/4v^2)(phi^2 - v^2)^2
    //
    // with Lambda = mu^2 / v^2, vacua at phi = +-v (v = vev), and V0 = mu^4/(4 Lambda)
    // chosen to zero the vacuum energy.  This is exactly the DWZ4 potential restricted
    // to the real axis (Lambda <-> lambda1 - 2 lambda2), so the kink width is the same
    // sqrt(2)/mu and the same (mu, fStar, omegaStar) convention applies.
    //
    // The point of Z2: a single real scalar has NO phase, so it cannot wind into a
    // global string and there are NO junctions -- only walls.  This is the clean
    // null-test for the PRS-vs-physical wall-area gap, which we attributed to junction
    // frustration in Z4.  If that gap is all junctions, Z2 phys/PRS -> 1.0; any residual
    // quantifies the junction-independent part (fat-wall smoothing + wall dynamics).
    //
    // PRS (fat wall + Hubble drag) is inherited generically from the base model
    // (prsWall / prsDamping / prsStartA) and the scalar-singlet kernel -- no model code.
    //
    // Output: one Scal column (adjacent = only wall type) in average_energies.txt, plus
    // a separate wall_diagnostics file carrying the potential-cutoff wall velocity
    // v^2 gamma^2 = sum(mask*KE)/sum(mask*V).
    /////////

    struct ModelPars : public TempLat::DefaultModelPars {
        static constexpr size_t NScalars   = 1;
        static constexpr size_t NPotTerms  = 1;
        // Z2: a single wall type (sign flip phi: +v <-> -v).
        static constexpr size_t NWallTypes = 1;
    };

  #define MODELNAME DWZ2

  template<class R>
  using Model = MakeModel(R, ModelPars);

  class MODELNAME : public Model<MODELNAME>
  {
  private:
    double mu, vev, Lambda;

  public:

    MODELNAME(ParameterParser& parser, RunParameters<double>& runPar,
              std::shared_ptr<MemoryToolBox> toolBox)
    : Model<MODELNAME>(parser, runPar.getLatParams(), toolBox, runPar.dt, STRINGIFY(MODELLABEL))
    {
        mu  = parser.get<double>("mu");
        vev = parser.get<double>("vev");
        Lambda = mu * mu / (vev * vev);            // V = -(mu^2/2)phi^2 + (Lambda/4)phi^4

        fldS0 = parser.get<double, 1>("initial_amplitudes");
        piS0  = parser.get<double, 1>("initial_momenta", {0});

        // Same rescaling convention as DWZ4 (real axis): fStar = v, omegaStar = mu.
        alpha     = 1;
        fStar     = vev;
        omegaStar = mu;

        setInitialPotentialAndMassesFromPotential();
    }

    /////////
    // Potential  (single term, vacuum-subtracted via +V0 = mu^4/(4 Lambda))
    /////////
    auto potentialTerms(Tag<0>)
    {
        return -(0.5) * mu * mu * pow<2>(fldS(0_c))
               + (0.25) * Lambda * pow<4>(fldS(0_c))
               + pow(mu, 4) / (4.0 * Lambda);
    }

    // dV/dphi = -mu^2 phi + Lambda phi^3
    auto potDeriv(Tag<0>)
    {
        return -mu * mu * fldS(0_c) + Lambda * pow<3>(fldS(0_c));
    }

    // d2V/dphi^2 = -mu^2 + 3 Lambda phi^2
    auto potDeriv2(Tag<0>)
    {
        return -mu * mu + 3.0 * Lambda * pow<2>(fldS(0_c));
    }

    /////////
    // Wall area: single sign-flip wall type (no junctions).
    /////////
    template<int WallIdx>
    auto wallAreaTerm(Tag<WallIdx>, double lSide_in, int N_in)
    {
        static_assert(WallIdx == 0, "Z2 has only one wall type (Tag<0>).");
        return FieldFunctionals::Z2vacuumSign(*this, lSide_in, N_in);
    }

    /////////
    // Wall-velocity diagnostic.  v^2 gamma^2 = sum(mask*KE)/sum(mask*V) over wall sites,
    // mask = potential cutoff V > alpha*Vmax (alpha = 0.2, 0.4, 0.6).  Mirrors the DWZ4
    // "selector A" velocity (single real field).  Written to a separate wall_diagnostics
    // file on the infrequent schedule; EnergiesMeasurer detects it via SFINAE.
    /////////

    // Per-site physical kinetic-energy density (0.5 a^-6 pi^2).
    auto wallKineticDensity()
    {
        return 0.5 * pow<-6>(aI) * pow<2>(piS(0_c));
    }

    // Per-site (vacuum-subtracted) potential-energy density.
    auto wallPotentialDensity()
    {
        return potentialTerms(Tag<0>());
    }

    std::vector<std::string> wallDiagnosticsHeaders() const
    {
        return { "velA02_KE", "velA02_V", "velA02_N",
                 "velA04_KE", "velA04_V", "velA04_N",
                 "velA06_KE", "velA06_V", "velA06_N" };
    }

    std::vector<double> wallDiagnosticsValues(double /*lSide_in*/, int /*N_in*/)
    {
        std::vector<double> out;
        out.reserve(9);

        // Barrier height at phi = 0 (vacuum-subtracted): V(0) - V(v) = mu^4/(4 Lambda).
        const double Vmax = pow(mu, 4) / (4.0 * Lambda);
        auto KE = wallKineticDensity();
        auto Vd = wallPotentialDensity();

        for (double a : {0.2, 0.4, 0.6}) {
            auto m = heaviside(Vd - a * Vmax);
            out.push_back(scale(m * KE));
            out.push_back(scale(m * Vd));
            out.push_back(scale(m));
        }
        return out;
    }

    };
}

#endif // DWZ2_H
