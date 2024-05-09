%module openmm_ic

%include "factory.i"
%import(module="openmm") "swig/OpenMMSwigHeaders.i"
%include "swig/typemaps.i"

/*
 * The following lines are needed to handle std::vector.
 * Similar lines may be needed for vectors of vectors or
 * for other STL types like maps.
 */

%include "std_vector.i"
namespace std {
    %template(vectord) vector<double>;
    %template(vectori) vector<int>;
};

%{
#include "OpenMM.h"
#include "OpenMMAmoeba.h"
#include "OpenMMDrude.h"
#include "OpenMMIC.h"
#include "openmm/RPMDIntegrator.h"
#include "openmm/RPMDMonteCarloBarostat.h"
%}

%pythoncode %{
    import openmm.unit as unit
%}

/*
 * Add units to function outputs.
 */
%pythonappend ICPlugin::ICLangevinIntegrator::getTemperature() const %{
    val = unit.Quantity(val, unit.kelvin)
%}

%pythonappend ICPlugin::ICLangevinIntegrator::getFriction() const %{
    val = unit.Quantity(val, 1 / unit.picosecond)
%}

%pythonappend ICPlugin::ICDrudeLangevinIntegrator::getTemperature() const %{
    val = unit.Quantity(val, unit.kelvin)
%}

%pythonappend ICPlugin::ICDrudeLangevinIntegrator::getFriction() const %{
    val = unit.Quantity(val, 1 / unit.picosecond)
%}

%pythonappend ICPlugin::ICDrudeLangevinIntegrator::getDrudeTemperature() const %{
    val = unit.Quantity(val, unit.kelvin)
%}

%pythonappend ICPlugin::ICDrudeLangevinIntegrator::getDrudeFriction() const %{
    val = unit.Quantity(val, 1 / unit.picosecond)
%}

%pythonappend ICPlugin::ICDrudeLangevinIntegrator::getMaxDrudeDistance() const %{
    val = unit.Quantity(val, unit.nanometer)
%}

/*
 * Convert C++ exceptions to Python exceptions.
 */
%exception {
    try {
        $action
    } catch (std::exception &e) {
        PyErr_SetString(PyExc_Exception, const_cast<char *>(e.what()));
        return NULL;
    }
}

namespace ICPlugin {

class ICLangevinIntegrator : public OpenMM::Integrator {
   public:
    ICLangevinIntegrator(double temperature, double frictionCoeff,
                         double stepSize, int numCells = 2,
                         double cellZSize = -1);

    double getTemperature() const;
    void setTemperature(double temp);
    double getFriction() const;
    void setFriction(double coeff);
    int getRandomNumberSeed() const;
    void setRandomNumberSeed(int seed);
    virtual void step(int steps);
    int getNumCells() const;
    void setNumCells(int cells);
    double getCellZSize() const;
    void setCellZSize(double cellZSize);
};

class ICDrudeLangevinIntegrator : public OpenMM::Integrator {
   public:
    ICDrudeLangevinIntegrator(double temperature, double friction,
                              double drudeTemperature, double drudeFriction,
                              double stepSize, int numCells = 2,
                              double cellZSize = -1);

    double getTemperature() const;
    void setTemperature(double temp);
    double getFriction() const;
    void setFriction(double coeff);
    double getDrudeTemperature() const;
    void setDrudeTemperature(double temp);
    double getDrudeFriction() const;
    void setDrudeFriction(double coeff);
    double getMaxDrudeDistance() const;
    void setMaxDrudeDistance(double distance);
    virtual void step(int steps);
    int getNumCells() const;
    void setNumCells(int cells);
    double getCellZSize() const;
    void setCellZSize(double cellZSize);
};

}  // namespace ICPlugin