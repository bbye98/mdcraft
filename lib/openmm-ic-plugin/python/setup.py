from distutils.core import Extension, setup
import os
import platform

openmm_dir = '@OPENMM_DIR@'
ic_plugin_header_dir = '@ICPLUGIN_HEADER_DIR@'
ic_plugin_library_dir = '@ICPLUGIN_LIBRARY_DIR@'

# setup extra compile and link arguments on Mac
extra_compile_args = []
extra_link_args = []

if platform.system() == 'Darwin':
    extra_compile_args += ['-stdlib=libc++', '-mmacosx-version-min=10.7']
    extra_link_args += ['-stdlib=libc++', '-mmacosx-version-min=10.7', '-Wl',
                        '-rpath', os.path.join(openmm_dir, 'lib')]

extension = Extension(name='_openmm_ic',
                      sources=['ICPluginWrapper.cpp'],
                      libraries=['OpenMM', 'OpenMMDrude', 'OpenMMIC'],
                      include_dirs=[os.path.join(openmm_dir, 'include'), 
                                    ic_plugin_header_dir],
                      library_dirs=[os.path.join(openmm_dir, 'lib'), 
                                    ic_plugin_library_dir],
                      extra_compile_args=extra_compile_args,
                      extra_link_args=extra_link_args)

setup(name='openmm-ic', version='1.0', py_modules=['openmm_ic'],
      ext_modules=[extension])