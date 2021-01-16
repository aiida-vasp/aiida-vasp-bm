import numpy as np
from aiida.manage.configuration import load_profile
from aiida.orm import Bool, Str, Code, Int, Float
from aiida.plugins import DataFactory, WorkflowFactory
from aiida.engine import submit

load_profile()

Dict = DataFactory('dict')
KpointsData = DataFactory("array.kpoints")


def launch_aiida_bulk_modulus(structure, code_string, resources,
                              label="AlN VASP relax calculation"):
    incar_dict = {'incar': {
        'PREC': 'Accurate',
        'EDIFF': 1e-8,
        'NELMIN': 5,
        'NELM': 100,
        'ENCUT': 500,
        'IALGO': 38,
        'ISMEAR': 0,
        'SIGMA': 0.01,
        'GGA': 'PS',
        'LREAL': False,
        'LCHARG': False,
        'LWAVE': False,}
    }

    kpoints = KpointsData()
    kpoints.set_kpoints_mesh([6, 6, 4], offset=[0, 0, 0.5])

    options = {'resources': resources,
               'max_wallclock_seconds': 3600 * 10}

    potential_family = 'PBE.54'
    potential_mapping = {'Al': 'Al', 'N': 'N'}

    parser_settings = {'add_energies': True,
                       'add_forces': True,
                       'add_stress': True}

    code = Code.get_from_string(code_string)
    Workflow = WorkflowFactory('vasp_bm.bulkmodulus')
    builder = Workflow.get_builder()
    builder.code = code
    builder.parameters = Dict(dict=incar_dict)
    builder.structure = structure
    builder.settings = Dict(dict={'parser_settings': parser_settings})
    builder.potential_family = Str(potential_family)
    builder.potential_mapping = Dict(dict=potential_mapping)
    builder.kpoints = kpoints
    builder.options = Dict(dict=options)
    builder.metadata.label = label
    builder.metadata.description = label
    builder.clean_workdir = Bool(False)
    relax = {}
    relax['perform'] = Bool(True)
    relax['force_cutoff'] = Float(1e-8)
    relax['steps'] = Int(10)
    relax['positions'] = Bool(True)
    relax['shape'] = Bool(True)
    relax['volume'] = Bool(True)
    relax['convergence_on'] = Bool(True)
    relax['convergence_volume'] = Float(1e-8)
    relax['convergence_max_iterations'] = Int(2)
    builder.relax = relax
    builder.verbose = Bool(True)

    node = submit(builder)
    return node


def get_structure_AlN():
    """Set up AlN primitive cell

     Al N
       1.0
         3.1100000000000000    0.0000000000000000    0.0000000000000000
        -1.5550000000000000    2.6933390057696038    0.0000000000000000
         0.0000000000000000    0.0000000000000000    4.9800000000000000
     Al N
       2   2
    Direct
       0.3333333333333333  0.6666666666666665  0.0000000000000000
       0.6666666666666667  0.3333333333333333  0.5000000000000000
       0.3333333333333333  0.6666666666666665  0.6190000000000000
       0.6666666666666667  0.3333333333333333  0.1190000000000000

    """

    StructureData = DataFactory('structure')
    a = 3.11
    c = 4.98
    lattice = [[a, 0, 0],
               [-a / 2, a / 2 * np.sqrt(3), 0],
               [0, 0, c]]
    structure = StructureData(cell=lattice)
    for pos_direct, symbol in zip(
            ([1. / 3, 2. / 3, 0],
             [2. / 3, 1. / 3, 0.5],
             [1. / 3, 2. / 3, 0.619],
             [2. / 3, 1. / 3, 0.119]), ('Al', 'Al', 'N', 'N')):
        pos_cartesian = np.dot(pos_direct, lattice)
        structure.append_atom(position=pos_cartesian, symbols=symbol)
    return structure


def main(code_string, resources):
    structure = get_structure_AlN()
    node = launch_aiida_bulk_modulus(structure, code_string, resources,
                                     label="AlN VASP calc")
    print(node)


if __name__ == '__main__':
    # code_string = 'vasp544mpi@gpu'
    # resources = {'parallel_env': 'mpi*', 'tot_num_mpiprocs': 12}
    code_string = 'vasp544mpi@nancy'
    resources = {'parallel_env': 'mpi*', 'tot_num_mpiprocs': 24}
    main(code_string, resources)
