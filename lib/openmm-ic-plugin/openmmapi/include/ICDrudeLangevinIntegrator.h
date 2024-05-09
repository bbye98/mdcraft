#ifndef OPENMM_ICDRUDELANGEVININTEGRATOR_H_
#define OPENMM_ICDRUDELANGEVININTEGRATOR_H_

#include "internal/windowsExportIC.h"
#include "openmm/DrudeKernels.h"
#include "openmm/Integrator.h"
#include "openmm/Kernel.h"

namespace ICPlugin {

class OPENMM_EXPORT_IC ICDrudeLangevinIntegrator : public OpenMM::Integrator {
   public:
    /**
     * Create a ICDrudeLangevinIntegrator.
     *
     * @param temperature           the temperature of the main heat bath (in
     * Kelvin)
     * @param frictionCoeff         the friction coefficient which couples the
     * system to the main heat bath (in inverse picoseconds)
     * @param drudeTemperature      the temperature of the heat bath applied to
     * internal coordinates of Drude particles (in Kelvin)
     * @param drudeFrictionCoeff    the friction coefficient which couples the
     * system to the heat bath applied to internal coordinates of Drude
     * particles (in inverse picoseconds)
     * @param stepSize              the step size with which to integrator the
     * system (in picoseconds)
     * @param numCells              the number of real and image cells
     * @param cellZSize             the z-dimension of the unit cell (in
     * nanometers)
     */
    ICDrudeLangevinIntegrator(double temperature, double frictionCoeff,
                              double drudeTemperature,
                              double drudeFrictionCoeff, double stepSize,
                              int numCells = 2, double cellZSize = -1);

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
     * Get the temperature of the heat bath applied to internal coordinates of
     * Drude particles.
     *
     * @return the temperature of the heat bath (in Kelvin)
     */
    double getDrudeTemperature() const { return drudeTemperature; }

    /**
     * Set the temperature of the heat bath applied to internal coordinates of
     * Drude particles.
     *
     * @param temp  the temperature of the heat bath (in Kelvin)
     */
    void setDrudeTemperature(double temp) {
        if (temp < 0)
            throw OpenMM::OpenMMException("Temperature cannot be negative");
        drudeTemperature = temp;
    }

    /**
     * Get the friction coefficient which determines how strongly the internal
     * coordinates of Drude particles are coupled to the heat bath.
     *
     * @return the friction coefficient (in inverse picoseconds)
     */
    double getDrudeFriction() const { return drudeFriction; }

    /**
     * Set the friction coefficient which determines how strongly the internal
     * coordinates of Drude particles are coupled to the heat bath.
     *
     * @param coeff    the friction coefficient (in inverse picoseconds)
     */
    void setDrudeFriction(double coeff) {
        if (coeff < 0)
            throw OpenMM::OpenMMException(
                "Friction coefficient cannot be negative");
        drudeFriction = coeff;
    }

    /**
     * Get the maximum distance a Drude particle can ever move from its parent
     * particle. This is implemented with a hard wall constraint.  If this
     * distance is set to 0 (the default), the hard wall constraint is omitted.
     *
     * @return the maximum distance a Drude particle can ever move from its
     * parent particle (in nanometers)
     */
    double getMaxDrudeDistance() const { return maxDrudeDistance; }

    /**
     * Set the maximum distance a Drude particle can ever move from its parent
     * particle. This is implemented with a hard wall constraint. If this
     * distance is set to 0 (the default), the hard wall constraint is omitted.
     *
     * @param distance    the maximum distance a Drude particle can ever move
     * from its parent particle (in nanometers)
     */
    void setMaxDrudeDistance(double distance) {
        if (distance < 0)
            throw OpenMM::OpenMMException("Distance cannot be negative");
        maxDrudeDistance = distance;
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
    void cleanup() {
        kernel = OpenMM::Kernel();
    }

    /**
     * Get the names of all Kernels used by this Integrator.
     */
    std::vector<std::string> getKernelNames();

    /**
     * Compute the kinetic energy of the system at the current time.
     */
    double computeKineticEnergy();

   private:
    double temperature, friction, drudeTemperature, drudeFriction,
        maxDrudeDistance, cellZSize;
    int randomNumberSeed, numCells;
    OpenMM::Kernel kernel;
};

}  // namespace ICPlugin

#endif /*OPENMM_ICDRUDELANGEVININTEGRATOR_H_*/