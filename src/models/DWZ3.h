#ifndef LPHI4_H //Usual macro guard to prevent multiple inclusion
#define LPHI4_H

/* This file is part of CosmoLattice, available at www.cosmolattice.net .
   Copyright Daniel G. Figueroa, Adrien Florio, Francisco Torrenti and Wessel Valkenburg.
   Released under the MIT license, see LICENSE.md. */

// File info: Main contributor(s): Daniel G. Figueroa, Adrien Florio, Francisco Torrenti,  Year: 2020

#include "CosmoInterface/cosmointerface.h"

//Include cosmointerface to have access to all of the library.

namespace TempLat
{
    /////////
    // Model name and number of fields
    /////////

    // In the following class, we define the defining parameters of your model:
    // number of fields of each species and the type of tinteractions.

    struct ModelPars : public TempLat::DefaultModelPars {
    	static constexpr size_t NScalars = 2;
        // In our phi4 example, we only want 2 scalar fields.
        static constexpr size_t NPotTerms = 3;
        // Our potential naturaly splits into two terms: the inflaton potential
        // and the interaction with the daughter field.

        // All the numbers of fields are 0 by default, so we need only
        // to specify that we want two scalar fields.
        // See the model with gauge fields to have an example of how to turn
        // them on and specify interactions.
    };

  #define MODELNAME DWZ3
  // Here we define the name of the model. This should match the name of your file.

  template<class R>
  using Model = MakeModel(R, ModelPars);
  // In this line, we define an appropriate generic model, with the correct
  // number of fields, ready to be customized.
  // If you are curious about what this is doing, the macro is defined in
  // the "CosmoInterface/abstractmodel.h" file.

  class MODELNAME : public Model<MODELNAME>
  // Declaration of our model. It inherits from the generic model defined above.
  {
 //...
private:

    double lambda1, lambda2, mu; //here the only parameters are the self interacting coupling lambda and the vev.
// Here are the declaration of the model specific parameters. They are 'private'
// to force you using them only within your model and not outside.
///SCALE FACTOR model.aI or model.aSI
// Some parameters which are declared in the class "Model" and which are useful (they are all 'public'):

// fldS0, piS0 : arrays which should contain the initial homogeneous values of
//               the scalar fields
//
// alpha, fStar, omegaStar : time and field rescaling to go to program units.
//
// fldS : The actual object which contains the scalar fields.

  public:

    MODELNAME(ParameterParser& parser, RunParameters<double>& runPar, std::shared_ptr<MemoryToolBox> toolBox): //Constructor of our model.
    Model<MODELNAME>(parser,runPar.getLatParams(), toolBox, runPar.dt, STRINGIFY(MODELLABEL)) //MODELLABEL is defined in the cmake.
    {

      /////////
      // Independent parameters of the model (read from parameters file)
      /////////

      lambda1 = parser.get<double>("lambda1");
      lambda2 = parser.get<double>("lambda2");
      //  We start by initializing our model paramteters. We read them from the
      // input file/command line.  Effectively, by calling 'par.get<double>("lambda")'
      // we declare a new parameter which needs to be in the input data.  Its name is
      // "lambda" and we specify it is a 'double'.

      mu  = parser.get<double>("mu");
      // In the same way, we declare an input parameter 'q'.
	 

      // g = sqrt(q*lambda);
      //For convenience, we also define g as a function of lambda and q.


        /////////
        // Initial homogeneous components of the fields
        // (read from parameters file, or specified here if not)
        /////////

        fldS0 = parser.get<double,2>("initial_amplitudes"); 
        piS0 = parser.get<double, 2>("initial_momenta", {0,0});
        
        // Then, we need to specify the initial homogeneous
        // value of our fields. We read them again from the input file. The int '2' means
        // that we actually expect two values and that we will get an array of
        // double of size two.
        // Contrary to the "initial_amplitudes" parameter and the others above,
        //, the "initial_momenta" is an optional parameter. It can still be specified through
        //  command line or input file as initial_momenta=value1 value2 ... valueNs,
        // but it can also be omitted, as we specified a default value of '{0, 0}'.


        /////////
        // Rescaling for program variables
        /////////
        double beta = 3*lambda2/(sqrt(8*lambda1));
        alpha = 1;
        fStar = mu/(2*sqrt(lambda1))*(beta+ sqrt(pow<2>(beta)+1)); //proper vacuum 
        omegaStar =mu;  // sqrt(lambda) * fStar;
        //my guess, it should work but I need to remember this, results might be weird
        //without proper interpretation
        // We now need to specify the rescaling from physical units to program units.
        // This consists of the  time rescaling exponent alpha, the field rescaling fStar
        // and the velocity rescaling omegaStar.
        // See the paper for more information on how to fix them.

        setInitialPotentialAndMassesFromPotential();
        // Here we call this function to compute the value of the potential on the homogeneous
        // initial condition  (useful to set the initial Hubble rate). We also compute
        // in this function the masses from the second derivative of the potential
        // evaluated on the homogeneous initial conditions. If you want to do something else,
        // uncomment the next section and do whatever suits your needs.

        /*
          masses2S = {..., ...};
          setInitialPotentialFromPotential();
         */
    }

   /////////
   // Program potential (add as many functions as terms are in the potential)
   /////////
   ///////// fldS(0_c) is h and fldS(1_c) is a 
    auto potentialTerms(Tag<0>) // Inflaton potential energy
    //
    // Now we need to define the physics of the model. We start by defining the potential.
    // We need to specify as  many potential  terms as we specified in the ModelParams,
    // here 2. Then for every potential terms, we define a function
    //' auto potentialTerms(Tag<N>)'  with N =0,...,NPot -1. The type 'Tag<N>' simply allows
    // to define different function with the same name. The 'auto' keyword lets the compiler
    // figure out on itself what is the actual return type of the function.
    {
      return -(0.5)*mu*mu*(pow<2>(fldS(0_c)) +pow<2>(fldS(1_c)) )  + (1/(512*pow(lambda1,3))*pow(3*lambda2+sqrt(lambda1)*sqrt(8+9*pow(lambda2,2)/lambda1),2)*(12*lambda1+3*(-3+4*sqrt(2))*pow(lambda2,2)+(-3+4*sqrt(2))*sqrt(lambda1)*lambda2*sqrt(8+9*pow(lambda2,2)/lambda1))*pow(mu,4) ) ;
    }

	// 0.25 * pow<4>(fldS(0_c));
        // Some notations.  The scalar fields are stored in a collection called 'fldS'.
        // The scalar fields are labelled  from 0 to Ns-1. The field say number 1 is
        // accessed through the syntax 'fldS(0_c)'. The function 'pow<N>(x)'. Works with the
        // known-at-compile-time integer N and compute the expression x*...*x N times.
        // If you don't know the integer at compile time or you don't have an integer,
        // use the more usual syntax pow(x, N).
        // These 'pow' functions are just one example of the many algebraic functions which
        // can be applied to our fields,  see the manual for an exhaustive list
        // and what to do if you want to implement a new one.
    
    // auto potentialTerms(Tag<1>) // Interaction energy
    // {
    //  return 0.5 * q * pow<2>(fldS(0_c) * fldS(1_c));
    // }
    auto potentialTerms(Tag<1>) 
    {
        return (0.25)*lambda1*pow<2>((pow<2>(fldS(0_c)) + pow<2>(fldS(1_c)) )) ;
    }
     auto potentialTerms(Tag<2>) 
    {
        return -(lambda2*mu/sqrt(2))*fldS(0_c)*(pow<2>(fldS(0_c)) - 3*pow<2>(fldS(1_c)) );
    }


	
    // Advanced note (ignore if you are satisfied with the above) :
    // - The 'auto' return type is important because the object returned is
    // not say an array containing  the value of the expression but the expression itself, which can and will be
    // evaluate later on. The type of the  expression itself depends on the expression and can be intricated. See
    // manual for more  details.
    // - The syntax 0_c is equivalent to Tag<0>(),
    // i.e. creating  an object of type 0. This operator '_c' is a modern C++ user-defined type literal,
    // taken from Boost and located in fcn/util/rangeiteration/tagliteral.h .



   /////////
   // Derivatives of the program potential with respect fields
   // (add one function for each field).
   /////////

    auto potDeriv(Tag<0>) // Derivative with respect to h.
    // In exactly the same fashion, we  need to define one derivative of the potential
    // per scalar field (2 in this case).  The integer in Tag<0> tells you the field with
    // respect to which you are defining the derivative of the potential of.
    {
      return   lambda1*fldS(0_c)* (pow<2>(fldS(0_c)) +pow<2>(fldS(1_c)) ) + (3*lambda2*mu/sqrt(2))*(fldS(1_c)-fldS(0_c))*(fldS(1_c)+fldS(0_c)) - mu*mu*fldS(0_c);
      //  pow<3>(fldS(0_c)) + q * fldS(0_c) * pow<2>(fldS(1_c)) ;
    }

    // auto potDeriv(Tag<1>)  // Derivative with respect to the daughter field.
    // {
    // return  q * fldS(1_c) * pow<2>(fldS(0_c));
    // }
 
	auto potDeriv(Tag<1>) // Derivative with respect to a
    // In exactly the same fashion, we  need to define one derivative of the potential
    // per scalar field (2 in this case).  The integer in Tag<0> tells you the field with
    // respect to which you are defining the derivative of the potential of.
    {
      return lambda1*fldS(1_c)* (pow<2>(fldS(0_c)) +pow<2>(fldS(1_c)) ) +mu*fldS(1_c)*(3*(lambda2*sqrt(2))*fldS(0_c)-mu) ;
      //  pow<3>(fldS(0_c)) + q * fldS(0_c) * pow<2>(fldS(1_c)) ;
    }
	
    /////////
   //  Second derivatives of the program potential with respect fields
   // (add one function for each field)
   /////////
	// da qua
  auto potDeriv2(Tag<0>) // Second derivative h
  {
    // d^2V/dS0^2 = 4*lambda1*(3*S0^2 + S1^2) - mu*(12*lambda2*S0 + 2)
    return lambda1*pow<2>(fldS(1_c)) + 3*lambda1*pow<2>(fldS(0_c))-mu*(3*sqrt(2)*lambda2*fldS(0_c) + mu);
  }

  auto potDeriv2(Tag<1>) // Second derivative wrt a
  {
    // d^2V/dS1^2 = 4*lambda1*(S0^2 + 3*S1^2) - mu*(2 - 12*lambda2*S0)
    return 3 *lambda1*pow<2>(fldS(1_c)) + lambda1*pow<2>(fldS(0_c)) + mu*(3*sqrt(2)*lambda2*fldS(0_c) - mu);
  }

		
	
    };
}

#endif //LPHI4_H
