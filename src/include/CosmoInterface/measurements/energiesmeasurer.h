#ifndef COSMOINTERFACE_MEASUREMENTS_ENERGIESMEASURER_H
#define COSMOINTERFACE_MEASUREMENTS_ENERGIESMEASURER_H
 
/* This file is part of CosmoLattice, available at www.cosmolattice.net .
   Copyright Daniel G. Figueroa, Adrien Florio, Francisco Torrenti and Wessel Valkenburg.
   Released under the MIT license, see LICENSE.md. */ 
   
// File info: Main contributor(s): Daniel G. Figueroa, Adrien Florio, Francisco Torrenti,  Year: 2020

#include "CosmoInterface/runparameters.h"
#include "CosmoInterface/measurements/meansmeasurer.h"
#include "CosmoInterface/measurements/measurementsIO/filesmanager.h"
#include "TempLat/util/templatvector.h"
#include "TempLat/util/rangeiteration/sum_in_range.h"
#include "TempLat/util/rangeiteration/tagliteral.h"
#include "TempLat/lattice/algebra/helpers/getngrid.h"
#include "CosmoInterface/definitions/energies.h"
#include "CosmoInterface/definitions/hubbleconstraint.h"
#include "CosmoInterface/definitions/gaugederivatives.h"
#include "CosmoInterface/definitions/fieldfunctionals.h"
#include <type_traits>


namespace TempLat {

    // Detect whether a model defines wall (junction/velocity) diagnostics.
    // Models without the method compile fine and simply skip the extra file.
    template<class M, class = void>
    struct HasWallDiagnostics : std::false_type {};
    template<class M>
    struct HasWallDiagnostics<M, std::void_t<decltype(std::declval<M&>().wallDiagnosticsValues(0.0, 0))>>
        : std::true_type {};

    /** \brief A class which contains measurements of energies and scale factor.
     *
     **/

    template <typename T>
    class EnergiesMeasurer {
    public:
        /* Put public methods here. These should change very little over time. */
        template <typename Model>
        EnergiesMeasurer(Model& model, FilesManager& filesManager, const RunParameters<T>& par, bool append) :
                amIRoot( model.getToolBox()->amIRoot()),
                lSide(par.lSide),
                N(par.N), 
                expansion(par.expansion),
                fixedBackground(par.fixedBackground), // boolean: if true, expansion is given by fixed background
                Etot0(0),  // Initial total energy
                energies(filesManager, "energies", amIRoot, append, getEnergyHeaders(model)),  // Output file for volume-average energies.
                energyCons(filesManager,  "energy_conservation", amIRoot, append, getEnergyConsHeaders(), fixedBackground),
                 // Output file for checking energy conservation.
                wallDiag(filesManager, "wall_diagnostics", amIRoot, append, getWallDiagHeaders(model), !HasWallDiagnostics<Model>::value)
                 // Output file for junction-string + wall-velocity diagnostics (only created if the model defines them).
                {
        }

        // Junction-string and wall-velocity diagnostics. Called on the INFREQUENT
        // schedule (transcendental-heavy); no-op for models without the method.
        template <class Model>
        void measureWallDiagnostics(Model& model, T t)
        {
            if constexpr (HasWallDiagnostics<Model>::value) {
                wallDiag.addAverage(t);
                for (double val : model.wallDiagnosticsValues(lSide, N)) wallDiag.addAverage(T(val));
                wallDiag.save();
            }
        }

        template <class Model>
        void measure(Model& model, T t, bool saveEtot)
        {
            energies.addAverage(t);  // add to file
            T Etot = 0;  // stores total energy
            T Egrad = 0;  // auxiliary variable, stores grad energy
            T Ekin = 0;  // auxiliary variable, stores kinetic energy
          

            // The "energies" functions contain the appropriate scale factor rescaling. Here we compute the energy species by species

            // Scalar singlets
            ForLoop(i,0, Model::Ns-1,
                    Ekin = average(Energies::kineticS(model,FieldFunctionals::pi2S(model,i)));
                    Egrad =  average(Energies::gradientS(model,FieldFunctionals::grad2S(model,i)));
                    Etot += Ekin + Egrad;  // add to total energy
                    energies.addAverage(Ekin);
                    energies.addAverage(Egrad);
            );
            
            // Complex scalars
            ForLoop(i,0, Model::NCs-1,
                    Ekin = average(Energies::kineticCS(model,FieldFunctionals::pi2CS(model,i)));
                    Egrad =  average(Energies::gradientCS(model,FieldFunctionals::grad2CS(model,i)));
                    Etot += Ekin + Egrad;   // add to total energy
                    energies.addAverage(Ekin);
                    energies.addAverage(Egrad);
            );
            
            // SU2 Doublets
            ForLoop(i,0, Model::NSU2Doublet-1,
                    Ekin = average(Energies::kineticSU2Doublet(model, FieldFunctionals::pi2SU2Doublet(model,i)));
                    Egrad =  average(Energies::gradientSU2Doublet(model,FieldFunctionals::grad2SU2Doublet(model,i)));
                    Etot += Ekin + Egrad;  // add to total energy
                    energies.addAverage(Ekin);
                    energies.addAverage(Egrad);
            );

            // U1 gauge fields
            ForLoop(i,0, Model::NU1-1,
                    Ekin = average(Energies::electricU1(model,FieldFunctionals::pi2U1(model,i)));
                    Egrad =  average(Energies::magneticU1(model,FieldFunctionals::B2U1(model,i)));
                    Etot += Ekin + Egrad;  // add to total energy
                    energies.addAverage(Ekin);
                    energies.addAverage(Egrad);
            );

			// SU2 gauge fields
            ForLoop(i,0, Model::NSU2-1,
                    Ekin = average(Energies::electricSU2(model,FieldFunctionals::pi2SU2(model,i)));
                    Egrad = average(Energies::magneticSU2(model,FieldFunctionals::B2SU2(model,i)));
                    Etot += Ekin + Egrad;  // add to total energy
                    energies.addAverage(Ekin);
                    energies.addAverage(Egrad);
            );
            
            // Potential
            T potTerm = 0;
            ForLoop(i, 0, Model::NPotTerms-1,
                    potTerm = average(model.potentialTerms(i));
                    energies.addAverage(potTerm);
                    Etot += potTerm;
            );
            energies.addAverage(Etot);
            // Write one area-parameter column per wall type (ξ_k = a A_k / L³).
            // Models that don't define NWallTypes (or set it to 0) skip this block entirely.
            if constexpr (Model::NWallTypes > 0) {
                ForLoop(wt, 0, Model::NWallTypes - 1,
                    energies.addAverage(scale(model.wallAreaTerm(wt, lSide, N)));
                );
            }
            energies.save();

            if(!fixedBackground) {  // Energy cannot be checked if expansion is fixed
            
                // We now check energy conservation:
                energyCons.addAverage(t);
                if (saveEtot) Etot0 = Etot;  // Saves the initial total energy before the first iteration

                if (expansion) {   // If self-consistent expansion, energy conservation is checked via the first Friedmann equation
                    auto hubbleLaw = HubbleConstraint::get(model);
                    energyCons.addAverage(hubbleLaw[0]);
                    energyCons.addAverage(hubbleLaw[1]);
                    energyCons.addAverage(hubbleLaw[2]);
                } 
                else {  // If no expansion, energy must be approximately constant during the evolution
                    energyCons.addAverage(abs(1.0 - Etot / Etot0));
                }

                energyCons.save();
            }
        }

    private:
    
     	// Returns string with the header of the energies file.
        template <typename Model>
        std::vector<std::string> getEnergyHeaders(Model& model) const 
        {
            std::vector<std::string> ret;
            ret.emplace_back("t");
            ForLoop(i,0, Model::Ns-1,
                    ret.emplace_back("E^kin_scal" + std::to_string(i));
                            ret.emplace_back("E^grad_scal" + std::to_string(i));
            );
            ForLoop(i,0, Model::NCs-1,
                    ret.emplace_back("E^kin_cmplxscal" + std::to_string(i));
                            ret.emplace_back("E^grad_cmplxscal" + std::to_string(i));
            );
            ForLoop(i,0, Model::NSU2Doublet-1,
                    ret.emplace_back("E^kin_SU2matter");
                            ret.emplace_back("E^grad_SU2matter");
            );
            ForLoop(i,0, Model::NU1-1,
                    ret.emplace_back("E^kin_U1" + std::to_string(i));
                            ret.emplace_back("E^grad_U1" + std::to_string(i));
            );
            ForLoop(i,0, Model::NSU2-1,
                    ret.emplace_back("E^kin_SU2");
                            ret.emplace_back("E^grad_SU2");
            );
            ForLoop(i,0, Model::NPotTerms-1,
                    ret.emplace_back("Vpot_term_" + std::to_string(i) );
            );

            ret.emplace_back("E_tot");
            // One column per wall type: Scal_type_1 (adjacent), Scal_type_2 (diagonal), …
            if constexpr (Model::NWallTypes > 0) {
                ForLoop(wt, 0, Model::NWallTypes - 1,
                    ret.emplace_back("Scal_type_" + std::to_string(int(wt) + 1));
                );
            }

            return ret;
        }

        // Returns header for the wall-diagnostics file ("t" + model's labels).
        template <typename Model>
        std::vector<std::string> getWallDiagHeaders(Model& model) const
        {
            std::vector<std::string> ret;
            ret.emplace_back("t");
            if constexpr (HasWallDiagnostics<Model>::value) {
                for (const auto& h : model.wallDiagnosticsHeaders()) ret.emplace_back(h);
            }
            return ret;
        }

		// Returns header for energy conservation file.
        std::vector<std::string> getEnergyConsHeaders() const
        {
            std::vector<std::string> ret;
            ret.emplace_back("t");
            if(expansion){
                ret.emplace_back("rel_diff_friedmann");
                ret.emplace_back("LHS_friedmann");
                ret.emplace_back("RHS_friedmann");
            }else{
                ret.emplace_back("energy_cons");
            }

            return ret;
        }

        /* Put all member variables and private methods here. These may change arbitrarily. */
        /* Put all member variables and private methods here. These may change arbitrarily. */
        const bool amIRoot;
        T lSide;
        int N;
        const bool expansion, fixedBackground;
        T Etot0;

        MeasurementsSaver<T> energies;
        MeasurementsSaver<T> energyCons;
        MeasurementsSaver<T> wallDiag;

    };

    class EnergiesMeasurerTester{
    public:
#ifdef TEMPLATTEST
        static inline void Test(TDDAssertion& tdd);
#endif
    };



} /* TempLat */

#ifdef TEMPLATTEST
#include "CosmoInterface/measurements/energiesmeasurer_test.h"
#endif


#endif
