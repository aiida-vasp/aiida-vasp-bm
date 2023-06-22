import numpy as np
from aiida.common.extendeddicts import AttributeDict
from aiida.manage.configuration import load_profile
from aiida.orm import Bool, Str, Code, Int, Float, WorkChainNode, QueryBuilder, Group
from aiida.plugins import DataFactory, WorkflowFactory
from aiida.engine import submit
load_profile()

Dict = DataFactory('dict')
KpointsData = DataFactory("array.kpoints")

def launch_aiida_bulk_modulus(structure, code_string, options,
                              label="VASP bulk modulus calculation"):
    incar_dict = {
        'incar': {
            'PREC': 'Accurate',
            'EDIFF': 1e-8,
            'NELMIN': 5,
            'NELM': 100,
            'ENCUT': 500,
            'IALGO': 38,
            'ISMEAR': 0,
            'SIGMA': 0.01,
            'LREAL': False,
            'LCHARG': False,
            'LWAVE': False
        }
    }

    kpoints = KpointsData()
    kpoints.set_kpoints_mesh([6, 6, 4], offset=[0, 0, 0.5])

    potential_family = 'PBE.54'
    potential_mapping = {'Si': 'Si', 'C': 'C'}

    parser_settings = {'add_energies': True,
                       'add_forces': True,
                       'add_stress': True}

    code = Code.get_from_string(code_string)
    Workflow = WorkflowFactory('vasp.bm')
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
    relax = AttributeDict()
    relax.perform = Bool(True)
    relax.force_cutoff = Float(1e-8)
    relax.steps = Int(100)
    relax.positions = Bool(True)
    relax.shape = Bool(True)
    relax.volume = Bool(True)
    builder.relax = relax
    builder.verbose = Bool(True)

    node = submit(builder)
    return node


def get_structure_SiC():
    """Set up SiC wurtzite cell

    wurtzite-type SiC
      1.0000000000
      3.0920000000   0.0000000000   0.0000000000
     -1.5460000000   2.6777505485   0.0000000000
      0.0000000000   0.0000000000   5.0730000000
    Si    C
        2     2
    Direct
      0.3333333333   0.6666666667   0.0000000000
      0.6666666667   0.3333333333   0.5000000000
      0.3333333333   0.6666666667   0.3758220000
      0.6666666667   0.3333333333   0.8758220000

    """

    StructureData = DataFactory('structure')
    a = 3.092
    c = 5.073
    lattice = [[a, 0, 0],
               [-a / 2, a / 2 * np.sqrt(3), 0],
               [0, 0, c]]
    structure = StructureData(cell=lattice)
    for pos_direct, symbol in zip(
            ([1. / 3, 2. / 3, 0],
             [2. / 3, 1. / 3, 0.5],
             [1. / 3, 2. / 3, 0.375822],
             [2. / 3, 1. / 3, 0.875822]), ('Si', 'Si', 'C', 'C')):
        pos_cartesian = np.dot(pos_direct, lattice)
        structure.append_atom(position=pos_cartesian, symbols=symbol)
    return structure

def main(code_string, options):
    structure = get_structure_SiC()
    node = launch_aiida_bulk_modulus(structure, code_string, options,
                                     label="SiC VASP bulk modulus calculation")
    print('Launched workchain node: ', node)

if __name__ == '__main__':
    # Code_string is chosen from output of the list given by 'verdi code list'
    code_string = 'vasp@mycluster'

    # Set the options
    options = {
        'resources': {
            'num_machines': 1,
            'num_mpiprocs_per_machine': 8
        },
        'account': '',
        'qos': '',
        'max_memory_kb': 2000000,
        'max_wallclock_seconds': 1800
    }

    # Run workflow
    main(code_string, options)
