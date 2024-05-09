#ifndef OPENMM_ICLANGEVININTEGRATOR_H_
#define OPENMM_ICLANGEVININTEGRATOR_H_

#include "internal/windowsExportIC.h"
#include "openmm/Integrator.h"
#include "openmm/Kernel.h"

namespace ICPlugin {

class OPENMM_EXPORT_IC ICLangevinIntegrator : public OpenMM::Integrator {
   public:
    /**
     * Create a ICLangevinIntegrator.
     *
     * @param temperature    the temperature of the heat bath (in Kelvin)
     * @param frictionCoeff  the friction coefficient which couples the system
     * to the heat bath (in inverse picoseconds)
     * @param stepSize       the step size with which to integrate the system
     * (in picoseconds)
     * @param numCells       the number of real and image cells
     * @param cellZSize      the z-dimension of the unit cell (in nanometers)
     */
    ICLangevinIntegrator(double temperature, double frictionCoeff,
                         double stepSize, int numCells = 2, double cellZSize = -1);

    /**
     * Get the temperature of the main heat bath.
     *
     * @return the temperature of the heat bath (in Kelvin)
     */
    double getTemperature() const { return temperature; }

    /**
     * Set the temperature of the main heat bath.

     * @param temp  the temperature of the heat bath (in Kelvin)
     */
    void setTemperature(double temp) {
        if (temp < 0)
            throw OpenMM::OpenMMException(
                "Temperature cannot be negative");
        temperature = temp;
    }

    /**
     * Get the friction coefficient which determines how strongly the system is
     * coupled to the main heat bath.
     *
     * @return the friction coefficient (in inverse picoseconds)
     */
    double getFriction() const { return friction; }

    /**
     * Set the friction coefficient which determines how strongly the system is
     * coupled to the main heat bath.
     *
     * @param coeff the friction coefficient (in inverse picoseconds)
     */
    void setFriction(double coeff) {
        if (coeff < 0)
            throw OpenMM::OpenMMException(
                "Friction coefficient cannot be negative");
        friction = coeff;
    }

    /**
     * Get the random number seed. See setRandomNumberSeed() for details.
     *
     * @return the random number seed
     */
    int getRandomNumberSeed() const { return randomNumberSeed; }

    /**
     * Set the random number seed.  The precise meaning of this parameter is
     * undefined, and is left up to each Platform to interpret in an appropriate
     * way.  It is guaranteed that if two simulations are run with different
     * random number seeds, the sequence of random forces will be different.  On
     * the other hand, no guarantees are made about the behavior of simulations
     * that use the same seed. In particular, Platforms are permitted to use
     * non-deterministic algorithms which produce different results on
     * successive runs, even if those runs were initialized identically.
     *
     * If seed is set to 0 (which is the default value assigned), a unique seed
     * is chosen when a Context is created from this Force. This is done to
     * ensure that each Context receives unique random seeds without you needing
     * to set them explicitly.
     *
     * @param seed    the random number seed
     */
    void setRandomNumberSeed(int seed) { randomNumberSeed = seed; }

    /**
     * Advance a simulation through time by taking a series of time steps.
     *
     * @param steps   the number of time steps to take
     */
    void step(int steps);

    /**
     * Get the number of real and image cells.
     *
     * @return the number of real and image cells
     */
    int getNumCells() const { return numCells; }

    /**
     * Set the number of real and image cells.
     *
     * @param numCells  the number of real and image cells
     */
    void setNumCells(int cells) { numCells = cells; }

    /**
     * Get z-dimension of the unit cell.
     *
     * @return the z-dimension of the unit cell (in nanometers)
     */
    double getCellZSize() const { return cellZSize; }

    /**
     * Set z-dimension of the unit cell.
     *
     * @param cellZSize the z-dimension of the unit cell (in nanometers)
     */
    void setCellZSize(double size) { cellZSize = size; }

   protected:
    /**
     * This will be called by the Context when it is created. It informs the
     * Integrator of what context it will be integrating, and gives it a chance
     * to do any necessary initialization. It will also get called again if the
     * application calls reinitialize() on the Context.
     */
    void initialize(OpenMM::ContextImpl& context);

    /**
     * This will be called by the Context when it is destroyed to let the
     * Integrator do any necessary cleanup. It will also get called again if
     * the application calls reinitialize() on the Context.
     */
    void cleanup() { kernel = OpenMM::Kernel(); }

    /**
     * Get the names of all Kernels used by this Integrator.
     */
    std::vector<std::string> getKernelNames();

    /**
     * Compute the kinetic energy of the system at the current time.
     */
    double computeKineticEnergy();

   private:
    double temperature, friction, cellZSize;
    int randomNumberSeed, numCells;
    OpenMM::Kernel kernel;
};

}  // namespace ICPlugin

#endif /*OPENMM_ICLANGEVININTEGRATOR_H_*/