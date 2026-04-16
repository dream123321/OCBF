import os
import sys
import time
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np

from ase.io import read, iread, write
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.data import atomic_numbers
from tqdm import tqdm
from loguru import logger

# 检查pymlip是否可用
try:
    from pymlip.core import MTPCalactor, PyConfiguration

    PYMLIP_AVAILABLE = True
except ImportError:
    PYMLIP_AVAILABLE = False
    logger.warning("pymlip未安装，SUS2计算器将不可用")

# 定义可用的计算器类型
CALCULATORS = {
    'nep': 'NEP',
    'mace': 'MACECalculator',
    'chgnet': 'CHGNetCalculator',
    'dp': 'DP',
    'm3gnet': 'PESCalculator',
    'mattersim': 'MatterSimCalculator',
    'sus2': 'SUS2Calculator',  # 新增SUS2计算器
}

WORKER_CALCULATOR = None


def configure_logger(log_level: str):
    """配置主进程或子进程日志。"""
    logger.remove()
    logger.add(sys.stderr, level=log_level)


def limit_blas_threads():
    """多进程 CPU 计算时，限制每个 worker 的底层线程数，避免过度抢占。"""
    for env_name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "BLIS_NUM_THREADS",
    ):
        os.environ.setdefault(env_name, "1")


class SUS2Calculator(Calculator):
    """
    SUS2 calculator based on ase Calculator
    基于MTP的SUS2势函数计算器
    """
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self,
                 potential: str = "p.sus2",
                 ele_list: Optional[List[str]] = None,
                 compute_stress: bool = True,
                 stress_weight: float = 1.0,
                 print_EK: bool = True,
                 **kwargs):
        """
        Args:
            potential (str): xxx.sus2 或 xxx.mtp 势函数文件
            ele_list (List[str]): 元素符号列表，例如 ["Al", "O"]
            compute_stress (bool): 是否计算应力
            stress_weight (float): 应力权重因子
            print_EK (bool): 是否打印能量信息
            **kwargs:
        """
        if not PYMLIP_AVAILABLE:
            raise ImportError("pymlip 未安装。请先安装 pymlip 以支持SUS2计算器")

        super().__init__(**kwargs)
        self.potential = potential
        self.compute_stress = compute_stress
        self.print_EK = print_EK
        self.stress_weight = stress_weight
        self.mtpcalc = MTPCalactor(self.potential)

        if ele_list is None:
            raise ValueError("SUS2计算器需要指定元素列表 (ele_list)")
        self.unique_numbers = [atomic_numbers[ele] for ele in ele_list]

        logger.info(f"SUS2计算器初始化完成")
        logger.info(f"势函数文件: {self.potential}")
        logger.info(f"元素列表: {ele_list} (原子序数: {self.unique_numbers})")

    def calculate(
            self,
            atoms: Optional[Atoms] = None,
            properties: Optional[list] = None,
            system_changes: Optional[list] = None,
    ):
        """
        Args:
            atoms (ase.Atoms): ase Atoms对象
            properties (list): 需要计算的属性列表
            system_changes (list): 监测原子系统的变化
        """
        properties = properties or ["energy"]
        system_changes = system_changes or self.all_changes
        super().calculate(atoms=atoms, properties=properties,
                          system_changes=system_changes)

        # 转换为pymlip配置
        cfg = PyConfiguration.from_ase_atoms(atoms, unique_numbers=self.unique_numbers)
        V = atoms.cell.volume if atoms.cell.volume > 0 else 1.0

        # 执行SUS2计算
        self.mtpcalc.calc(cfg)

        # 获取能量和力
        energy = np.array(cfg.energy)
        forces = cfg.force

        self.results['energy'] = energy
        self.results['forces'] = forces

        # 计算应力（如果需要）
        if self.compute_stress and hasattr(cfg, 'stresses') and cfg.stresses is not None:
            try:
                # SUS2/MTP输出的应力是完整张量，转换为Voigt记法
                stresses = cfg.stresses
                # Voigt记法: [xx, yy, zz, yz, xz, xy]
                self.results['stress'] = -np.array([
                    stresses[0, 0],  # xx
                    stresses[1, 1],  # yy
                    stresses[2, 2],  # zz
                    stresses[1, 2],  # yz
                    stresses[0, 2],  # xz
                    stresses[0, 1]  # xy
                ]) * self.stress_weight / V
            except (IndexError, AttributeError) as e:
                logger.debug(f"应力计算失败: {e}")
                pass


def scf(stru, calculator):
    """单点能量计算函数

    Args:
        stru: ASE原子结构
        calculator: ASE计算器实例

    Returns:
        atoms: 包含计算结果的Atoms对象
    """
    stru.calc = calculator
    e = stru.get_potential_energy()
    f = stru.get_forces()

    # 创建新的Atoms对象
    atoms = Atoms(stru.get_chemical_symbols(),
                  positions=stru.get_positions(),
                  cell=stru.get_cell())

    # 存储能量和力
    atoms.info['energy'] = e
    atoms.arrays['forces'] = f

    # 尝试获取应力
    try:
        s = stru.get_stress(voigt=False)  # 先尝试获取完整张量
        if s is not None and len(s) == 9:
            atoms.info['stress'] = s
            atoms.info['stress_GPa'] = s * 160.21766208  # 转换为GPa
            virial = -1 * s * stru.get_volume()
            atoms.info['virial'] = virial
        else:
            # 如果get_stress返回的是Voigt记法，转换为完整张量
            s_voigt = stru.get_stress(voigt=True)
            six2nine = np.array([s_voigt[0], s_voigt[5], s_voigt[4],
                                 s_voigt[5], s_voigt[1], s_voigt[3],
                                 s_voigt[4], s_voigt[3], s_voigt[2]])
            atoms.info['stress'] = six2nine
            atoms.info['stress_GPa'] = six2nine * 160.21766208
            virial = -1 * six2nine * stru.get_volume()
            atoms.info['virial'] = virial
    except (NotImplementedError, AttributeError) as e:
        logger.debug(f"应力计算不被支持或失败: {e}")
        pass

    atoms.info['pbc'] = "T T T"
    atoms.pbc = [True, True, True]

    return atoms


def setup_calculator(calc_type, model_path=None, device='cpu', ele_list=None):
    """设置计算器

    Args:
        calc_type: 计算器类型 ('nep', 'mace', 'chgnet', 'dp', 'm3gnet', 'mattersim', 'sus2')
        model_path: 模型文件路径
        device: 计算设备 ('cpu'或'cuda')
        ele_list: 元素列表 (仅对SUS2计算器必需)

    Returns:
        calculator: ASE计算器实例
    """

    if calc_type == 'nep':
        from pynep.calculate import NEP
        if not model_path:
            raise ValueError("NEP计算器需要指定模型文件路径")
        return NEP(model_path)

    elif calc_type == 'mace':
        try:
            from mace.calculators import MACECalculator
        except ImportError:
            raise ImportError("请安装MACE包: pip install mace-torch")

        if not model_path:
            raise ValueError("MACE计算器需要指定模型文件路径")
        return MACECalculator(model_paths=model_path, device=device)

    elif calc_type == 'dp':
        try:
            from deepmd.calculator import DP
        except ImportError:
            raise ImportError("请安装DeePMD包: pip install deepmd-kit")

        if not model_path:
            raise ValueError("DP计算器需要指定模型文件路径")
        return DP(model=model_path, device=device)

    elif calc_type == 'chgnet':
        try:
            from chgnet.model import CHGNet
            from chgnet.model.dynamics import CHGNetCalculator

            if model_path:
                model = CHGNet.from_file(model_path)
            else:
                # 使用默认模型
                model = CHGNet.load()

            return CHGNetCalculator(model, use_device=device)
        except ImportError:
            raise ImportError("请安装CHGNet包: pip install chgnet")

    elif calc_type == 'm3gnet':
        try:
            import matgl
            matgl.set_backend("DGL")
            from matgl.ext.ase import PESCalculator

            if model_path:
                model = matgl.load_model(model_path)
            else:
                # 使用预训练模型
                model = matgl.load_model("M3GNet-MP-2021.2.8-PES")

            return PESCalculator(model)
        except ImportError:
            raise ImportError("请安装matgl包: pip install matgl")

    elif calc_type == 'mattersim':
        try:
            from mattersim.forcefield import MatterSimCalculator

            logger.info(f"初始化MatterSim计算器，使用设备: {device}")

            if model_path:
                calculator = MatterSimCalculator(
                    load_path=model_path,
                    device=device
                )
            else:
                calculator = MatterSimCalculator(device=device)

            return calculator

        except ImportError:
            raise ImportError("请安装MatterSim包: pip install mattersim")
        except Exception as e:
            raise ImportError(f"MatterSim初始化失败: {e}")

    elif calc_type == 'sus2':
        if not PYMLIP_AVAILABLE:
            raise ImportError("pymlip未安装，无法使用SUS2计算器")

        if not model_path:
            raise ValueError("SUS2计算器需要指定势函数文件路径")

        if ele_list is None:
            raise ValueError("SUS2计算器需要指定元素列表，请使用 --ele_list 参数")

        logger.info(f"初始化SUS2计算器")
        logger.info(f"势函数文件: {model_path}")
        logger.info(f"元素列表: {ele_list}")

        return SUS2Calculator(
            potential=model_path,
            ele_list=ele_list,
            compute_stress=True
        )

    else:
        raise ValueError(f"不支持的计算器类型: {calc_type}. 可选: {list(CALCULATORS.keys())}")


def load_structures(input_path: Path) -> List[Atoms]:
    """读取输入结构；对 xyz/extxyz 显式预读取到内存，便于并行计算。"""
    suffix = input_path.suffix.lower()

    if suffix in {'.xyz', '.extxyz'}:
        logger.info("检测到 xyz/extxyz 输入，预先读取全部结构到内存")
        structures = read(input_path, index=':')
        if isinstance(structures, Atoms):
            return [structures]
        return list(structures)

    logger.info("读取输入结构到内存")
    return list(iread(input_path))


def resolve_num_workers(requested_workers: int) -> int:
    """解析用户指定的并行进程数。"""
    available_cpus = os.cpu_count() or 1

    if requested_workers <= 0:
        return available_cpus

    if requested_workers > available_cpus:
        logger.warning(f"请求的 CPU 核数 {requested_workers} 超过可用核数 {available_cpus}，将使用 {available_cpus}")
        return available_cpus

    return requested_workers


def init_worker(calc_type: str,
                model_path: Optional[str],
                device: str,
                ele_list: Optional[List[str]],
                log_level: str):
    """为每个进程初始化独立计算器，避免跨进程序列化 calculator。"""
    global WORKER_CALCULATOR
    limit_blas_threads()
    configure_logger(log_level)
    WORKER_CALCULATOR = setup_calculator(
        calc_type=calc_type,
        model_path=model_path,
        device=device,
        ele_list=ele_list
    )


def process_structure(task: Tuple[int, Atoms]):
    """子进程执行单个结构的静态计算。"""
    global WORKER_CALCULATOR
    index, structure = task

    if WORKER_CALCULATOR is None:
        raise RuntimeError("worker 计算器未初始化")

    try:
        atoms = scf(structure, WORKER_CALCULATOR)
        return index, atoms, None
    except Exception as exc:
        return index, None, str(exc)


def write_result(output_file: Path, atoms: Atoms, output_format: str, append_mode: bool):
    """统一写出计算结果。"""
    write(output_file, atoms, format=output_format, append=append_mode)


def main():
    parser = argparse.ArgumentParser(
        description='使用不同机器学习势函数进行结构能量计算',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python calc.py input.xyz --calc_type nep --model model.txt
  python calc.py input.extxyz --calc_type mace --model model.pth --device cuda
  python calc.py input.cif --calc_type chgnet --output results
  python calc.py input.xyz --calc_type mattersim --device cuda
  python calc.py input.xyz --calc_type sus2 --model model.sus2 --ele_list Al O  # 使用SUS2计算器
  python calc.py input.xyz --calc_type sus2 --model model.mtp --ele_list Al     # 使用MTP格式的势函数
  python calc.py input.xyz --calc_type mace --suffix test                       # 添加自定义后缀
        """
    )

    parser.add_argument('input', help='输入结构文件')
    parser.add_argument('--calc_type', '-c',
                        choices=list(CALCULATORS.keys()),
                        default='sus2',
                        help='计算器类型 (默认: sus2)')
    parser.add_argument('--model', '-m',
                        help='模型/势函数文件路径')
    parser.add_argument('--ele_list', '-e',
                        nargs='+',
                        help='元素列表，例如: --ele_list Al O (仅对SUS2计算器必需)')
    parser.add_argument('--device', '-d',
                        default='cpu',
                        choices=['cpu', 'cuda'],
                        help='计算设备 (cpu或cuda，默认: cpu)')
    parser.add_argument('--output', '-o',
                        default='out_files',
                        help='输出目录 (默认: out_files)')
    parser.add_argument('--format', '-f',
                        default='extxyz',
                        help='输出格式 (默认: extxyz)')
    parser.add_argument('--append', '-a',
                        action='store_true',
                        help='追加到输出文件而不是覆盖')
    parser.add_argument('--log-level', '-l',
                        default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='日志级别 (默认: INFO)')
    parser.add_argument('--suffix', '-s',
                        default='',
                        help='在输出文件名中添加后缀 (例如: --suffix test 生成 input_test.xyz)')
    parser.add_argument('--num-workers', '-n',
                        type=int,
                        default=1,
                        help='CPU 并行进程数；1 为串行，0 或负数表示使用全部可用 CPU 核')
    parser.add_argument('--chunksize',
                        type=int,
                        default=1,
                        help='多进程任务分块大小 (默认: 1)')

    args = parser.parse_args()

    # 配置logger
    configure_logger(args.log_level)

    if args.chunksize < 1:
        logger.error("--chunksize 必须大于等于 1")
        sys.exit(1)

    # 检查输入文件
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"输入文件不存在: {args.input}")
        sys.exit(1)

    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 自动检测CUDA可用性
        if args.device == 'cuda':
            if args.calc_type == 'sus2':
                logger.warning("SUS2 仅支持 CPU，将使用 CPU 进行计算")
                args.device = 'cpu'
            else:
                try:
                    import torch
                    if torch.cuda.is_available():
                        logger.info(f"CUDA可用，使用GPU: {torch.cuda.get_device_name(0)}")
                    else:
                        logger.warning("未检测到可用 CUDA，将使用 CPU 进行计算")
                        args.device = 'cpu'
                except ImportError:
                    logger.warning("未能导入 torch 检查 CUDA，继续按用户指定设备初始化")

        args.num_workers = resolve_num_workers(args.num_workers)
        parallel_enabled = args.device == 'cpu' and args.num_workers > 1

        if args.device != 'cpu' and args.num_workers > 1:
            logger.warning("GPU 模式不启用多进程并行，已自动切换为单进程")
            args.num_workers = 1
            parallel_enabled = False

        if parallel_enabled:
            limit_blas_threads()
            logger.info(f"启用 CPU 并行计算，worker 数: {args.num_workers}，chunksize: {args.chunksize}")
        else:
            logger.info("使用单进程计算")

        calculator = None
        if not parallel_enabled:
            calculator = setup_calculator(
                args.calc_type,
                args.model,
                args.device,
                args.ele_list
            )

        logger.info(f"使用 {CALCULATORS[args.calc_type]} 计算器")
        if args.model:
            logger.info(f"模型文件: {args.model}")
        logger.info(f"计算设备: {args.device}")

    except Exception as e:
        logger.error(f"设置计算器失败: {e}")
        sys.exit(1)

    try:
        structures = load_structures(input_path)
        logger.info(f"读取到 {len(structures)} 个结构")
    except Exception as e:
        logger.error(f"读取输入文件失败: {e}")
        sys.exit(1)

    if len(structures) == 0:
        logger.warning("未读取到任何结构，程序结束")
        sys.exit(0)

    # 生成输出文件名
    stem = input_path.stem
    suffix = f"_{args.suffix}" if args.suffix else ""
    extension = input_path.suffix

    if args.format == input_path.suffix.lstrip('.'):
        output_filename = f"{args.calc_type}_{stem}{suffix}{extension}"
    else:
        if args.format == 'extxyz':
            output_filename = f"{args.calc_type}_{stem}{suffix}.xyz"
        else:
            output_filename = f"{args.calc_type}_{stem}{suffix}.{args.format}"

    output_file = output_dir / output_filename
    logger.info(f"输出文件: {output_file}")

    # 执行计算
    logger.info("开始计算...")
    start_time = time.time()

    successful = 0
    failed = 0
    append_mode = args.append

    if parallel_enabled:
        try:
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(
                    max_workers=args.num_workers,
                    mp_context=ctx,
                    initializer=init_worker,
                    initargs=(args.calc_type, args.model, args.device, args.ele_list, args.log_level)
            ) as executor:
                results = executor.map(
                    process_structure,
                    enumerate(structures),
                    chunksize=args.chunksize
                )

                for index, atoms, error in tqdm(results, total=len(structures), desc="计算进度"):
                    if error is not None:
                        logger.error(f"结构 {index + 1} 计算失败: {error}")
                        failed += 1
                        continue

                    try:
                        write_result(output_file, atoms, args.format, append_mode)
                        successful += 1
                        append_mode = True
                    except Exception as exc:
                        logger.error(f"结构 {index + 1} 写出失败: {exc}")
                        failed += 1

        except Exception as e:
            logger.error(f"并行计算失败: {e}")
            sys.exit(1)
    else:
        for i, stru in enumerate(tqdm(structures, desc="计算进度")):
            try:
                atoms = scf(stru, calculator)
                write_result(output_file, atoms, args.format, append_mode)
                successful += 1
                append_mode = True
            except Exception as e:
                logger.error(f"结构 {i + 1} 计算失败: {e}")
                failed += 1
                continue

    end_time = time.time()
    elapsed = (end_time - start_time) / 60

    # 输出统计信息
    logger.info("=" * 50)
    logger.info(f"计算完成!")
    logger.info(f"成功: {successful} 个结构")
    logger.info(f"失败: {failed} 个结构")
    logger.info(f"总时间: {elapsed:.2f} 分钟")
    if successful > 0:
        logger.info(f"平均每个结构: {elapsed * 60 / successful:.2f} 秒")
    logger.info("=" * 50)


if __name__ == '__main__':
    main()
